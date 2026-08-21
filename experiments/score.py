"""Score generations into the parquet, and train the moral scorer."""

import sys
import argparse
import glob
import json
import os
import pickle
import re
import pandas as pd
from experiments._args import add_limit, add_model, add_steering_tag
from src.data import MU_SOURCES, source_for_tag
from src.evaluation import METRIC_GROUPS, EvalSuite, score_rows
from src.paths import (
    basename, isc_dir, isc_item, model_dir, mu_injection_root, shift_injection_root,
)
from src.evaluation import read_injection_scores
from src.data import METRIC_COLUMNS, make_row, merge, read, results_path, to_frame, write
from src.variants import attribute_for, eval_data_type, eval_set_for
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.paths import MORAL_DIR, SCORER_DIR


def _cli_generations(argv):
    ISC_ITEM_RE = re.compile(r"(\d+)_(\d{5})\.pkl$")

    def mu_source_of(tag: str) -> str:
        """'steering_jigsaw' -> 'jigsaw'."""
        return source_for_tag(tag)

    def ingest_isc(model_name, variants, limit):
        model = basename(model_name)
        rows = []
        for variant in variants:
            split = "train" if variant.endswith("_train") else "test"
            base_variant = variant[: -len("_train")] if split == "train" else variant
            data_type = eval_data_type(base_variant)
            eval_set = eval_set_for(base_variant)
            d = isc_dir(model_name, base_variant, split)
            items = {}
            for path in sorted(glob.glob(os.path.join(d, "*.pkl"))):
                m = ISC_ITEM_RE.search(path)
                r, idx = int(m.group(1)), int(m.group(2))
                if limit and idx >= limit:
                    continue
                with open(path, "rb") as f:
                    rec = pickle.load(f)
                items[(r, idx)] = rec

            for (r, idx), rec in sorted(items.items()):
                ref = items.get((0, idx))
                stored = {k: rec[k] for k in METRIC_COLUMNS if k in rec}
                rows.append(make_row(
                    model=model, condition="isc", variant=base_variant, eval_set=eval_set,
                    data_type=data_type, split=split, round=r, item_idx=idx,
                    text=rec["result"],
                    _prefix=rec["prompt"],
                    _reference=ref["result"] if ref else rec["result"],
                    **stored,
                ))
        return rows

    def ingest_injection(model_name, tag, limit):
        model = basename(model_name)
        mu_source = mu_source_of(tag)
        attribute = MU_SOURCES[source_for_tag(tag)]["attribute"]
        rows = []
        roots = [(mu_injection_root(model_name, tag), "norm")]

        for root, mode in roots:
          for d in sorted(glob.glob(os.path.join(root, "*_*"))):
            data_type, alpha = os.path.basename(d).rsplit("_", 1)
            alpha = float(alpha)

            for path in sorted(glob.glob(os.path.join(d, "*.json"))):
                idx = int(os.path.basename(path)[:-5])
                if limit and idx >= limit:
                    continue
                with open(path) as f:
                    rec = json.load(f)
                base_text = rec["baseline_text"]

                common = dict(model=model, mu_source=mu_source, data_type=data_type, item_idx=idx,
                              variant=f"steer:{mode}",
                              _prefix=rec["prompt"], _reference=base_text)
                rows.append(make_row(
                    condition="baseline", alpha=0.0, text=base_text,
                    **read_injection_scores(attribute, rec, "baseline"),
                    **common,
                ))
                for key in [k for k in rec if k.startswith("steered_L") and k.endswith("_text")]:
                    layer = int(key[len("steered_L") : -len("_text")])
                    rows.append(make_row(
                        condition="steer", layer=layer, alpha=alpha, text=rec[key],
                        **read_injection_scores(attribute, rec, f"steered_L{layer}"),
                        **common,
                    ))
        return rows

    TRANSPLANT_DIR_RE = re.compile(r"^(?P<data_type>.+)_a(?P<alpha>[\d.]+)(?P<mode>n?)_L(?P<layer>\d+)$")

    def ingest_transplant(model_name, tag, variant, limit):
        """sweep_shift_injection rows."""
        model = basename(model_name)
        mu_source = mu_source_of(tag)
        rows, missing_ref = [], 0
        root = shift_injection_root(model_name, tag)

        for d in sorted(glob.glob(os.path.join(root, "*_L*"))):
            m = TRANSPLANT_DIR_RE.match(os.path.basename(d))
            if not m:
                continue
            data_type = m.group("data_type")

            for path in sorted(glob.glob(os.path.join(d, "*.json"))):
                idx = int(os.path.basename(path)[:-5])
                if limit and idx >= limit:
                    continue
                with open(path) as f:
                    rec = json.load(f)

                ref_path = isc_item(model_name, variant, 0, idx, "test")
                if not os.path.exists(ref_path):
                    missing_ref += 1
                    continue
                with open(ref_path, "rb") as f:
                    ref_text = pickle.load(f)["result"]

                mode = "norm" if m.group("mode") == "n" else "abs"
                rows.append(make_row(
                    model=model, mu_source=mu_source, condition="transplant",
                    variant=f"{variant}:{mode}",
                    data_type=data_type, item_idx=idx,
                    layer=int(rec["layer"]), alpha=float(rec["alpha"]),
                    text=rec["transplant_text"],
                    **read_injection_scores(attribute_for(variant), rec, "transplant"),
                    _prefix=rec["prompt"], _reference=ref_text,
                ))

        if missing_ref:
            print(f"WARNING: {missing_ref} transplant rows skipped -- no ISC round-0 reference. "
                  f"Run run_self_correction on --variants {variant} for these items, or their content-preservation "
                  f"columns would be meaningless.", flush=True)
        return rows

    _RESUME_KEY = ["condition", "variant", "data_type", "split", "round", "layer",
                   "alpha", "item_idx"]

    _GROUP_OF = {
        "roberta_tox": "tox", "detoxify_tox": "tox", "ppl": "ppl",
        "judge_fluency": "judge", "bertscore_f1": "sim", "distinct2": "distinct2",
        "cola_fluency": "cola",
        "moral_score": "moral", "rougeL_moral": "moral", "rougeL_immoral": "moral",
        "bleu_moral": "moral", "bleu_immoral": "moral",
    }

    _ungrouped = [c for c in METRIC_COLUMNS if c not in _GROUP_OF]

    if _ungrouped:
        raise RuntimeError(f"METRIC_COLUMNS entries missing from _GROUP_OF: {_ungrouped}")

    def keys_of(df):
        """Both sides of the resume check must be coerced identically: a row dict's `layer=None` and the parquet's `<NA>` are the same key, and comparing them raw silently re-scores every ISC row on every pass."""
        return df[_RESUME_KEY].astype(str).agg("|".join, axis=1)

    def already_scored(df, metrics):
        """Keys whose requested metric columns are all non-null in the existing parquet."""
        cols = [c for c in METRIC_COLUMNS if _GROUP_OF[c] in metrics]
        if df is None or df.empty or not cols:
            return set()
        return set(keys_of(df.dropna(subset=cols)))

    def summarise(df):
        g = df.groupby(["condition", "variant", "round", "layer", "alpha"],
                       dropna=False)[METRIC_COLUMNS]
        out = g.mean().round(3)
        out["n"] = g.size()
        return out

    def main(args):
        metrics = tuple(METRIC_GROUPS) if "all" in args.metrics else tuple(args.metrics)
        out_path = args.out or results_path(basename(args.model), args.results_dir)
        existing = read(out_path)

        rows = []
        if "isc" in args.ingest:
            variants = args.variants or [
                os.path.basename(p) for p in sorted(glob.glob(os.path.join(model_dir(args.model), "isc", "*")))
            ]
            rows += ingest_isc(args.model, variants, args.limit)
        if "injection" in args.ingest:
            rows += ingest_injection(args.model, args.steering_tag, args.limit)
        if "transplant" in args.ingest:
            rows += ingest_transplant(args.model, args.steering_tag, args.variant, args.limit)

        if not rows:
            raise SystemExit(f"nothing to ingest under {model_dir(args.model)}")

        new_df = to_frame(rows)
        done = set() if args.force else already_scored(existing, metrics)
        todo = [r for r, k in zip(rows, keys_of(new_df)) if k not in done]
        print(f"ingested {len(rows)} rows; {len(todo)} need {metrics}", flush=True)

        if todo:
            suite = EvalSuite(hf_token=None)
            score_rows(suite, todo, metrics=metrics, batch_size=args.batch_size)
            if "judge" in metrics:
                print(f"judge: {suite.judge_parse_failures}/{suite.judge_calls} replies unparsable",
                      flush=True)

        df = merge(existing, to_frame(rows))
        write(df, out_path)
        print(f"wrote {len(df)} rows -> {out_path}")

        if args.summary:
            pd.set_option("display.width", 200, "display.max_columns", 20)
            print(summarise(df))

    p = argparse.ArgumentParser()
    add_model(p)
    p.add_argument("--ingest", nargs="+", default=["isc", "injection"],
                   choices=["isc", "injection", "transplant"])
    p.add_argument("--variants", nargs="+", default=None,
                   help="default: every variant directory present on disk")
    p.add_argument("--variant", default="strong_detox",
                   help="the ISC variant the transplant's ell_bar was estimated from -- its "
                        "round-0 text is the transplant's baseline and BERTScore reference")
    add_steering_tag(p)
    p.add_argument("--metrics", nargs="+", default=["ppl", "sim", "distinct2"],
                   choices=list(METRIC_GROUPS) + ["all"],
                   help="'tox' re-scores what the stages already stored; 'judge' loads an 8B "
                        "model and is meant to be run on its own")
    add_limit(p)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--results_dir", default="results")
    p.add_argument("--out", default=None)
    p.add_argument("--force", action="store_true", help="re-score rows already populated")
    p.add_argument("--summary", action="store_true")
    main(p.parse_args(argv))


def _cli_train_moral_scorer(argv):
    BASE_MODEL = "roberta-large"

    MAX_SEQ_LENGTH = 100

    BATCH_SIZE = 16

    LEARNING_RATE = 1e-5

    MAX_EPOCHS = 50

    WARMUP_PCT = 0.1

    SEED = 42

    EVAL_EVERY = 500

    PATIENCE = 10

    ADAM_EPS = 1e-8

    MAX_GRAD_NORM = 1.0

    CKPT = os.path.join(SCORER_DIR, "moral_roberta")

    SPLITS = {
        "normal": "test_normal.jsonl",
        "lexical": "test_lexical.jsonl",
        "minimal": "test_minimal.jsonl",
    }

    def read(name):
        return pd.read_json(os.path.join(MORAL_DIR, "scorer", name), lines=True)

    def encode(tok, df):
        """<CLS> grounding <SEP> target <SEP> -- a segment pair, as in their utils.py."""
        enc = tok(list(df["grounding"]), list(df["target"]), truncation=True,
                  padding="max_length", max_length=MAX_SEQ_LENGTH, return_tensors="pt")
        return TensorDataset(enc["input_ids"], enc["attention_mask"],
                             torch.tensor(df["label"].to_numpy()))

    @torch.no_grad()
    def accuracy(model, loader, device):
        model.eval()
        correct = total = 0
        for ids, mask, labels in loader:
            logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
            correct += (logits.argmax(-1).cpu() == labels).sum().item()
            total += len(labels)
        model.train()
        return correct / max(total, 1)

    def train(args):
        from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                                  get_linear_schedule_with_warmup)

        torch.manual_seed(SEED)
        np.random.seed(SEED)
        device = args.device

        tok = AutoTokenizer.from_pretrained(BASE_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(
            BASE_MODEL, num_labels=2).to(device)

        tr, dv = read("train.jsonl"), read("dev.jsonl")
        print(f"train {len(tr)} rows ({tr['label'].mean():.3f} positive) | dev {len(dv)} rows")

        train_loader = DataLoader(encode(tok, tr), batch_size=BATCH_SIZE, shuffle=True,
                                  drop_last=True)
        dev_loader = DataLoader(encode(tok, dv), batch_size=BATCH_SIZE)

        total_steps = MAX_EPOCHS * len(train_loader)
        opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, eps=ADAM_EPS)
        sched = get_linear_schedule_with_warmup(
            opt, int(WARMUP_PCT * total_steps), total_steps)

        best, since_improved, step = -1.0, 0, 0
        os.makedirs(CKPT, exist_ok=True)
        model.train()

        for epoch in range(MAX_EPOCHS):
            for ids, mask, labels in train_loader:
                out = model(input_ids=ids.to(device), attention_mask=mask.to(device),
                            labels=labels.to(device))
                out.loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                opt.step()
                sched.step()
                opt.zero_grad()
                step += 1

                if step % EVAL_EVERY == 0:
                    acc = accuracy(model, dev_loader, device)
                    flag = ""
                    if acc > best:
                        best, since_improved, flag = acc, 0, "  <- best, saved"
                        model.save_pretrained(CKPT)
                        tok.save_pretrained(CKPT)
                    else:
                        since_improved += 1
                    print(f"  epoch {epoch} step {step}/{total_steps} "
                          f"loss {out.loss.item():.4f} dev_acc {acc:.4f}{flag}", flush=True)
                    if since_improved >= PATIENCE:
                        print(f"\nearly stop: {PATIENCE} evaluations without improvement "
                              f"(best dev acc {best:.4f})")
                        return
        print(f"\nreached the {MAX_EPOCHS}-epoch ceiling (best dev acc {best:.4f})")

    def evaluate(args):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if not os.path.isdir(CKPT):
            raise SystemExit(f"no checkpoint at {CKPT} -- run without --eval_only first")
        tok = AutoTokenizer.from_pretrained(CKPT)
        model = AutoModelForSequenceClassification.from_pretrained(CKPT).to(args.device).eval()

        report = {}
        for name, fname in SPLITS.items():
            df = read(fname)
            loader = DataLoader(encode(tok, df), batch_size=64)
            report[name] = {"accuracy": accuracy(model, loader, args.device), "n": int(len(df))}
            model.eval()

        with open(os.path.join(CKPT, "validation.json"), "w") as f:
            json.dump(report, f, indent=2)

        print("\n| split | n | accuracy |")
        print("|---|---|---|")
        for name in ("normal", "lexical", "minimal"):
            r = report[name]
            print(f"| {name} | {r['n']} | {r['accuracy']:.3f} |")
        print(f"\nwrote {os.path.join(CKPT, 'validation.json')}")

    p = argparse.ArgumentParser()
    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--device", type=str, default="cuda")
    a = p.parse_args(argv)
    if not a.eval_only:
        train(a)
    evaluate(a)


_COMMANDS = {"generations": _cli_generations, "train_moral_scorer": _cli_train_moral_scorer}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] not in _COMMANDS:
        sys.exit("usage: python -m experiments.score <cmd>  (cmd: " + ", ".join(_COMMANDS) + ")")
    _COMMANDS[argv[0]](argv[1:])


if __name__ == "__main__":
    main()
