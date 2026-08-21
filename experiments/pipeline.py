"""Data preparation, contrastive pairs, steering vectors, and the self-correction dialog."""

import sys
import argparse
import os
from src.model import _LAYER_PATHS
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from src.paths import RTP_DIR
import json
from src.data import allocate
from src.paths import ETHICS_DIR, MORAL_DIR
from src.data import MU_SOURCES
from src.paths import ETHICS_DIR, JIGSAW_DIR, MORAL_DIR, PROCESSED_DIR, RTP_DIR
from collections import defaultdict
import torch
from experiments._args import add_model
from src.model import load_model, num_layers
from src.paths import PROCESSED_DIR, classes_json, pair_file, steer_json, steering_tag, vector_dir
from src.model import POOLERS
from src.utils import seed_everything
import gc
import pickle
import numpy as np
from transformers import logging
from experiments._args import add_data_dir, add_generation, add_model
from src.data import select_eval_items
from src.paths import isc_dir, isc_item
from src.model import masked_mean
from src.evaluation import SCORERS
from src.variants import (
    PUBLISHED_VARIANTS, VARIANTS, attribute_for, eval_data_type, eval_set_for,
    round0_from_templates, system_prompt_for, templates_for,
)


def _cli_check_models(argv):
    def main(models):
        from transformers import AutoConfig

        token = os.environ.get("HF_TOKEN") or None
        if not token:
            print("   WARNING: HF_TOKEN is not set. Gated repos (Mistral, zephyr) will fail.")

        bad = []
        for m in models:
            try:
                cfg = AutoConfig.from_pretrained(m, token=token)
                n = getattr(cfg, "num_hidden_layers", None) or getattr(cfg, "n_layer", "?")
                d = getattr(cfg, "hidden_size", "?")
                print(f"   OK   {m:42s} {cfg.model_type:12s} L={n} d={d}")
            except Exception as e:
                first = str(e).strip().splitlines()[0][:100]
                print(f"   FAIL {m:42s} {type(e).__name__}: {first}")
                bad.append(m)

        if bad:
            print(f"\n   {len(bad)} model(s) unreachable. Fix before sweeping "
                  f"(usually: export HF_TOKEN, or accept the licence on the model page).")
            return 1

        print(f"\n   all {len(models)} models reachable. Decoder layers are found by probing "
              f"{len(_LAYER_PATHS)} layout conventions, so the architecture itself is not gated.")
        return 0

    sys.exit(main(argv))


def _cli_prepare_toxicity(argv):
    def create_splits(toxic_data, non_toxic_data, data_dir):
        os.makedirs(data_dir, exist_ok=True)

        tox_df = pd.read_json(toxic_data, lines=True)
        non_tox_df = pd.read_json(non_toxic_data, lines=True)

        tox_df["toxicity"] = tox_df["prompt"].apply(lambda d: d.get("toxicity") if isinstance(d, dict) else None)
        non_tox_df["toxicity"] = non_tox_df["prompt"].apply(lambda d: d.get("toxicity") if isinstance(d, dict) else None)

        tox_df["tox_bin"] = pd.qcut(tox_df["toxicity"], q=10, duplicates="drop")
        toxic_train, toxic_test = train_test_split(
            tox_df, test_size=1000, stratify=tox_df["tox_bin"], random_state=87
        )
        toxic_train = toxic_train.drop(columns=["tox_bin"])
        toxic_test = toxic_test.drop(columns=["tox_bin"])
        toxic_train.to_json(os.path.join(data_dir, "toxic_train_4k.jsonl"), orient="records", lines=True, force_ascii=False)
        toxic_test.to_json(os.path.join(data_dir, "toxic_test_1k.jsonl"), orient="records", lines=True, force_ascii=False)

        plt.figure(figsize=(9, 5.5))
        plt.hist(toxic_train["toxicity"], bins=10, alpha=0.55, label="Train")
        plt.hist(toxic_test["toxicity"], bins=10, alpha=0.55, label="Test")
        plt.title("Toxicity Distribution of toxic train and test splits")
        plt.xlabel("Toxicity")
        plt.ylabel("Density")
        plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(data_dir, "tox_splits_distribution.png"), dpi=160)
        plt.close()

        print("\nToxic splits toxicity stats:")
        print(f"Train split. mean = {toxic_train['toxicity'].mean()}, std = {toxic_train['toxicity'].std()}")
        print(f"Test  split. mean = {toxic_test['toxicity'].mean()}, std = {toxic_test['toxicity'].std()}")

        non_tox_df["tox_bin"] = pd.qcut(non_tox_df["toxicity"], q=10, duplicates="drop")
        non_toxic_train, non_toxic_test = train_test_split(non_tox_df, test_size=1000, stratify=non_tox_df["tox_bin"], random_state=87)
        non_toxic_train = non_toxic_train.drop(columns=["tox_bin"])
        non_toxic_test = non_toxic_test.drop(columns=["tox_bin"])
        non_toxic_train.to_json(os.path.join(data_dir, "non_toxic_train_4k.jsonl"), orient="records", lines=True, force_ascii=False)
        non_toxic_test.to_json(os.path.join(data_dir, "non_toxic_test_1k.jsonl"), orient="records", lines=True, force_ascii=False)

        plt.figure(figsize=(9, 5.5))
        plt.hist(non_toxic_train["toxicity"], bins=10, alpha=0.55, label="Train")
        plt.hist(non_toxic_test["toxicity"], bins=10, alpha=0.55, label="Test")
        plt.title("Toxicity Distribution of non-toxic train and test splits")
        plt.xlabel("Toxicity")
        plt.ylabel("Density")
        plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(data_dir, "non_tox_splits_distribution.png"), dpi=160)
        plt.close()

        print("\nNon-toxic splits toxicity stats:")
        print(f"Train split. mean = {non_toxic_train['toxicity'].mean()}, std = {non_toxic_train['toxicity'].std()}")
        print(f"Test  split. mean = {non_toxic_test['toxicity'].mean()}, std = {non_toxic_test['toxicity'].std()}")

    parser = argparse.ArgumentParser()
    parser.add_argument("--toxic_data", type=str,
                        default=os.path.join(RTP_DIR, "unprocessed_toxic_5k.jsonl"),
                        help="Path to the raw toxic JSONL file.")
    parser.add_argument("--non_toxic_data", type=str,
                        default=os.path.join(RTP_DIR, "unprocessed_non_toxic_5k.jsonl"),
                        help="Path to the raw non-toxic JSONL file.")
    parser.add_argument("--data_dir", type=str, default=RTP_DIR,
                        help="Directory where split files and plots are saved.")
    args = parser.parse_args(argv)
    create_splits(args.toxic_data, args.non_toxic_data, args.data_dir)


def _cli_prepare_moral(argv):
    REPO = "demelin/moral_stories"

    ETHICS_REPO = "hendrycks/ethics"

    CLS = "cls-action+context-{}"

    def load_full():
        """-> {ID: row} over the whole corpus, for the story fields."""
        from datasets import load_dataset

        ds = load_dataset(REPO, "full", split="train")
        return {row["ID"]: row for row in ds}

    def load_cls_ids(strategy, split):
        """-> list[str] of STORY ids in one official split."""
        from datasets import load_dataset

        ds = load_dataset(REPO, CLS.format(strategy), split=split)
        return [row["ID"][:-1] for row in ds]

    def story_text(row, moral: bool) -> str:
        action = row["moral_action"] if moral else row["immoral_action"]
        return f"{row['situation']} {row['intention']} {action}".strip()

    def prefix_text(row) -> str:
        return f"{row['situation']} {row['intention']}".strip()

    def write(path, records):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        root = MORAL_DIR if path.startswith(MORAL_DIR) else os.path.dirname(path)
        print(f"  {os.path.relpath(path, root):28s} {len(records):5d}")

    def prompt_record(text, score):
        return {"prompt": {"text": text, "toxicity": score}}

    def scorer_records(full, ids):
        """Two rows per story: the moral action (label 1) and the immoral one (label 0)."""
        out = []
        for sid in ids:
            row = full[sid]
            g = prefix_text(row)
            out.append({"ID": sid, "grounding": g, "target": row["moral_action"], "label": 1})
            out.append({"ID": sid, "grounding": g, "target": row["immoral_action"], "label": 0})
        return out

    ETHICS_MIN_CHARS = 50

    ETHICS_MAX_CHARS = 300

    def write_ethics(n=512):
        """ETHICS commonsense -> data/ethics/{moral,immoral}_mu.jsonl, balanced and length-matched."""
        from datasets import load_dataset

        ds = load_dataset(ETHICS_REPO, "commonsense", split="train")
        def keep(row):
            text = row["input"].strip()
            return ETHICS_MIN_CHARS <= len(text) <= ETHICS_MAX_CHARS

        rows = [r for r in ds if keep(r)]
        moral = [r["input"].strip() for r in rows if int(r["label"]) == 0]
        immoral = [r["input"].strip() for r in rows if int(r["label"]) == 1]

        moral_sorted = sorted(moral, key=len)
        immoral_sorted = sorted(immoral, key=len)
        k = min(len(moral_sorted), len(immoral_sorted), n)
        moral_sel = moral_sorted[:: max(len(moral_sorted) // k, 1)][:k]
        immoral_sel = immoral_sorted[:: max(len(immoral_sorted) // k, 1)][:k]

        write(os.path.join(ETHICS_DIR, "moral_mu.jsonl"),
              [prompt_record(t, 1.0) for t in moral_sel])
        write(os.path.join(ETHICS_DIR, "immoral_mu.jsonl"),
              [prompt_record(t, 0.0) for t in immoral_sel])

        mc = sum(map(len, moral_sel)) / max(k, 1)
        ic = sum(map(len, immoral_sel)) / max(k, 1)
        print(f"  ethics mean chars: moral={mc:.1f} immoral={ic:.1f} "
              f"gap={abs(mc - ic) / max(mc, ic):.1%}")

    def main(args):
        print(f"loading {REPO} ...")
        full = load_full()

        minimal = load_cls_ids("minimal_pairs", "test")
        lexical = load_cls_ids("lexical_bias", "test")
        norm = load_cls_ids("norm_distance", "test")
        norm_train = load_cls_ids("norm_distance", "train")
        norm_dev = load_cls_ids("norm_distance", "validation")
        print(f"  full={len(full)}  minimal_test={len(set(minimal))}  "
              f"lexical_test={len(set(lexical))}  norm_test={len(set(norm))}  "
              f"norm_train={len(set(norm_train))}  norm_dev={len(set(norm_dev))}")

        alloc = allocate(
            all_ids=list(full), minimal_ids=minimal, lexical_ids=lexical,
            norm_ids=norm, norm_train_ids=norm_train, norm_dev_ids=norm_dev,
            n_mu=args.n_mu, n_eval=args.n_eval, n_ellbar=args.n_ellbar,
        )

        print(f"writing to {MORAL_DIR}/")
        write(os.path.join(MORAL_DIR, "moral_mu.jsonl"),
              [prompt_record(story_text(full[i], True), 1.0) for i in alloc["mu"]])
        write(os.path.join(MORAL_DIR, "immoral_mu.jsonl"),
              [prompt_record(story_text(full[i], False), 0.0) for i in alloc["mu"]])
        write(os.path.join(MORAL_DIR, "prefix_eval.jsonl"),
              [prompt_record(prefix_text(full[i]), 1.0) for i in alloc["eval"]])
        write(os.path.join(MORAL_DIR, "prefix_train.jsonl"),
              [prompt_record(prefix_text(full[i]), 1.0) for i in alloc["ellbar"]])

        write(os.path.join(MORAL_DIR, "references.jsonl"), [
            {"prefix": prefix_text(full[i]),
             "moral_action": full[i]["moral_action"],
             "immoral_action": full[i]["immoral_action"]}
            for i in alloc["eval"] + alloc["ellbar"]
        ])

        sdir = os.path.join(MORAL_DIR, "scorer")
        write(os.path.join(sdir, "train.jsonl"), scorer_records(full, alloc["scorer_train"]))
        write(os.path.join(sdir, "dev.jsonl"), scorer_records(full, alloc["scorer_dev"]))
        write(os.path.join(sdir, "test_minimal.jsonl"), scorer_records(full, sorted(set(minimal))))
        write(os.path.join(sdir, "test_lexical.jsonl"), scorer_records(full, sorted(set(lexical))))
        write(os.path.join(sdir, "test_normal.jsonl"), scorer_records(full, sorted(set(norm))))

        def mean_chars(ids, moral):
            return sum(len(story_text(full[i], moral)) for i in ids) / max(len(ids), 1)

        m, im = mean_chars(alloc["mu"], True), mean_chars(alloc["mu"], False)
        print(f"\n  mean chars: moral={m:.1f}  immoral={im:.1f}  "
              f"gap={abs(m - im) / max(m, im):.1%}  (>30% means mu may be a length direction)")

        sets = {k: set(v) for k, v in alloc.items()}
        for a in sets:
            for b in sets:
                if a < b and sets[a] & sets[b]:
                    raise SystemExit(f"OVERLAP between {a} and {b}: {len(sets[a] & sets[b])} ids")
        print("  all allocations pairwise disjoint")

        if not args.skip_ethics:
            print(f"\nloading {ETHICS_REPO} (commonsense) -> {ETHICS_DIR}/ ...")
            write_ethics(args.n_mu)

    p = argparse.ArgumentParser()
    p.add_argument("--n_mu", type=int, default=512)
    p.add_argument("--n_eval", type=int, default=500)
    p.add_argument("--n_ellbar", type=int, default=200)
    p.add_argument("--skip_ethics", action="store_true",
                   help="rebuild data/moral/ without re-fetching the ETHICS transfer corpus")
    main(p.parse_args(argv))


def _cli_contrastive_pairs(argv):
    def post_process(neg_prompt_score, pos_prompt_score, high_score_slot):
        """Order each class by how strongly it exemplifies its own side."""
        if high_score_slot == "pos":
            pos_prompt_score = sorted(pos_prompt_score, key=lambda x: x[1], reverse=True)
            neg_prompt_score = sorted(neg_prompt_score, key=lambda x: x[1])
        elif high_score_slot == "neg":
            neg_prompt_score = sorted(neg_prompt_score, key=lambda x: x[1], reverse=True)
            pos_prompt_score = sorted(pos_prompt_score, key=lambda x: x[1])
        else:
            raise ValueError(f"high_score_slot must be 'pos' or 'neg', got {high_score_slot!r}")
        return neg_prompt_score, pos_prompt_score

    def save_pairs(source, neg_prompt_score, pos_prompt_score, classes, high_score_slot, output_dir):
        """Write the contrastive pairs plus the manifest that records mu's sign convention."""
        neg_prompt_score, pos_prompt_score = post_process(
            neg_prompt_score, pos_prompt_score, high_score_slot
        )

        out = os.path.join(output_dir, str(source))
        os.makedirs(out, exist_ok=True)

        for slot, pairs in (("neg", neg_prompt_score), ("pos", pos_prompt_score)):
            df = pd.DataFrame(
                [{"prompt": p, "toxicity": s} for p, s in pairs], columns=["prompt", "toxicity"]
            )
            df.to_json(
                os.path.join(out, f"{slot}_prompt_score.jsonl"),
                orient="records",
                lines=True,
                force_ascii=False,
            )

        with open(os.path.join(out, "classes.json"), "w") as f:
            json.dump({
                "pos": classes["pos"],
                "neg": classes["neg"],
                "mu": f"mean({classes['pos']}) - mean({classes['neg']})",
            }, f, indent=2)

        def mean_chars(pairs):
            return sum(len(p) for p, _ in pairs) / max(len(pairs), 1)

        print(f"  -> {out}: {len(neg_prompt_score)} {classes['neg']} / "
              f"{len(pos_prompt_score)} {classes['pos']}")
        print(f"     mean chars: {classes['neg']}={mean_chars(neg_prompt_score):.1f}  "
              f"{classes['pos']}={mean_chars(pos_prompt_score):.1f}  "
              f"(a large gap means mu may be a length direction -- see load_jigsaw)")

    def main(sources, output_dir, dirs, limit):
        for source in sources:
            spec = MU_SOURCES[source]
            data_dir = dirs.get(source) or spec["default_dir"]
            print(f"[{source}] loading from {data_dir}")
            kwargs = {"limit": limit} if limit else {}
            neg, pos = spec["loader"](data_dir, **kwargs)
            save_pairs(source, neg, pos, spec["classes"], spec["high_score_slot"], output_dir)

    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", nargs="+", default=["rtp"], choices=sorted(MU_SOURCES))
    parser.add_argument("--data_dir", type=str, default=RTP_DIR, help="RTP splits")
    parser.add_argument("--jigsaw_dir", type=str, default=JIGSAW_DIR, help="Jigsaw 2018 CSVs")
    parser.add_argument("--moral_dir", type=str, default=MORAL_DIR,
                        help="data/moral/, built by experiments.pipeline prepare_moral")
    parser.add_argument("--ethics_dir", type=str, default=ETHICS_DIR,
                        help="data/ethics/, built by experiments.pipeline prepare_moral")
    parser.add_argument("--output_dir", type=str, default=PROCESSED_DIR)
    parser.add_argument("--limit", type=int, default=None, help="cap pairs per source")
    args = parser.parse_args(argv)

    main(
        sources=args.sources,
        output_dir=args.output_dir,
        dirs={"rtp": args.data_dir, "jigsaw": args.jigsaw_dir,
              "moral": args.moral_dir, "ethics": args.ethics_dir},
        limit=args.limit,
    )


def _cli_steering_vectors(argv):
    DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}

    def assert_classes(source, processed_dir_root):
        """Fail loudly if the pairs on disk are not the classes this source claims."""
        if source not in MU_SOURCES:
            raise SystemExit(
                f"--sources {source!r} is not a MU_SOURCES key. Known sources: "
                f"{sorted(MU_SOURCES)}."
            )
        path = classes_json(source, processed_dir_root)
        if not os.path.exists(path):
            print(f"  [{source}] no classes.json (pre-manifest source); assuming "
                  f"mu = mean(pos) - mean(neg) as built")
            return
        manifest = json.load(open(path))
        spec = MU_SOURCES[source]
        if spec["classes"] != {"pos": manifest["pos"], "neg": manifest["neg"]}:
            raise SystemExit(
                f"classes.json for {source!r} says pos={manifest['pos']} neg={manifest['neg']}, "
                f"but MU_SOURCES[{source!r}] says {spec['classes']}. Re-run "
                f"experiments.pipeline contrastive_pairs."
            )
        print(f"  [{source}] mu = {manifest['mu']} (validated against MU_SOURCES[{source!r}])")

    def dataset_preprocessing(args):
        """Read the {prompt, toxicity} contract written by build_contrastive_pairs, honouring --data_ratio."""
        t_prompts, nt_prompts = [], []
        for source in args.sources:
            assert_classes(source, args.data_processed_dir)
            df_t = pd.read_json(
                pair_file(source, "neg", args.data_processed_dir), lines=True
            )
            df_nt = pd.read_json(
                pair_file(source, "pos", args.data_processed_dir), lines=True
            )
            n_t = int(len(df_t) * args.data_ratio)
            n_nt = int(len(df_nt) * args.data_ratio)
            t_prompts += df_t["prompt"].tolist()[:n_t]
            nt_prompts += df_nt["prompt"].tolist()[:n_nt]

        n = min(len(t_prompts), len(nt_prompts), args.limit)
        return t_prompts[:n], nt_prompts[:n]

    def corpus_max_len(tokenizer, texts, cap):
        longest = max(len(tokenizer.encode(t)) for t in texts)
        return min(longest, cap)

    def pooled_means(model, tokenizer, texts, args, n_layers, max_len):
        """Mean pooled hidden state per layer over `texts`."""
        pooler = POOLERS[args.pooling]
        legacy = args.pooling == "legacy_pad"
        device = next(model.parameters()).device

        totals = defaultdict(lambda: None)
        count = 0

        for start in range(0, len(texts), args.batch_size):
            batch = texts[start : start + args.batch_size]

            if legacy:
                enc = tokenizer(
                    batch, padding="max_length", max_length=max_len,
                    truncation=True, return_tensors="pt",
                )
                model_kwargs = {"input_ids": enc["input_ids"].to(device)}
                mask = None
            else:
                enc = tokenizer(
                    batch, padding=True, max_length=max_len,
                    truncation=True, return_tensors="pt",
                )
                model_kwargs = {k: v.to(device) for k, v in enc.items()}
                mask = enc["attention_mask"].to(device)

            with torch.no_grad():
                out = model(**model_kwargs, use_cache=False, output_hidden_states=True)

            for layer in range(n_layers):
                h = out.hidden_states[layer + 1].float()
                h = torch.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)
                pooled = pooler(h, mask).sum(dim=0).cpu()
                totals[layer] = pooled if totals[layer] is None else totals[layer] + pooled

            count += len(batch)
            del out
            print(f"  forward: {count}/{len(texts)}", flush=True)

        return {layer: totals[layer] / count for layer in range(n_layers)}, count

    def build(args):
        model, tokenizer = load_model(
            args.model,
            dtype=DTYPES[args.dtype],
            device_map=args.device_map,
        )
        n_layers = num_layers(model)

        t_prompts, nt_prompts = dataset_preprocessing(args)
        print(f"{len(t_prompts)} toxic / {len(nt_prompts)} non-toxic prompts, {n_layers} layers")

        max_len = (
            corpus_max_len(tokenizer, t_prompts + nt_prompts, args.max_len)
            if args.pooling == "legacy_pad"
            else args.max_len
        )

        print("toxic:")
        mean_t, n_t = pooled_means(model, tokenizer, t_prompts, args, n_layers, max_len)
        print("non-toxic:")
        mean_nt, n_nt = pooled_means(model, tokenizer, nt_prompts, args, n_layers, max_len)

        steering_vec = {
            layer: (mean_nt[layer] - mean_t[layer]).tolist() for layer in range(n_layers)
        }

        tag = steering_tag(args.sources, args.data_ratio, args.tag_suffix)
        out_dir = vector_dir(args.model, tag)
        os.makedirs(out_dir, exist_ok=True)
        with open(steer_json(args.model, tag), "w") as f:
            json.dump(steering_vec, f)

        meta = {
            "model": args.model,
            "sources": args.sources,
            "data_ratio": args.data_ratio,
            "pooling": args.pooling,
            "dtype": args.dtype,
            "max_len": max_len,
            "n_pairs": min(n_t, n_nt),
            "n_layers": n_layers,
            "d_model": len(steering_vec[0]),
        }
        with open(os.path.join(out_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        norms = [torch.tensor(v).norm().item() for v in steering_vec.values()]
        print(f"\nwrote {steer_json(args.model, tag)}")
        print(f"  d_model={meta['d_model']}  ||mu|| min={min(norms):.3f} max={max(norms):.3f}")

    def run():
        parser = argparse.ArgumentParser()
        add_model(parser)
        parser.add_argument("--sources", type=str, default=["rtp"], nargs="+",
                            help="MU_SOURCES keys, e.g. rtp jigsaw moral ethics")
        parser.add_argument("--data_ratio", type=float, default=1.0)
        parser.add_argument("--data_processed_dir", type=str, default=PROCESSED_DIR)
        parser.add_argument("--limit", type=int, default=20000000)
        parser.add_argument("--batch_size", type=int, default=16)
        parser.add_argument("--max_len", type=int, default=128,
                            help="token truncation cap; Jigsaw comments run far longer than RTP prefixes")
        parser.add_argument("--pooling", type=str, default="masked", choices=sorted(POOLERS))
        parser.add_argument("--tag_suffix", type=str, default="",
                            help="e.g. 'legacy' -> steering_rtp_legacy, so a variant run "
                                 "does not overwrite the real vectors")
        parser.add_argument("--dtype", type=str, default="bfloat16", choices=sorted(DTYPES),
                            help="bf16 by default: it leaves mu unchanged (cos=0.9999 vs fp32) and "
                                 "halves the weights, which is the difference between a 7B fitting "
                                 "on one 24GB card and not. See the module docstring.")
        parser.add_argument("--device_map", type=str, default="auto")
        args = parser.parse_args(argv)

        seed_everything(0, benchmark=True)
        build(args)

    run()


def _cli_self_correction(argv):
    logging.set_verbosity_error()

    def per_layer_response_mean(model, sequences, start_idx, pad_token_id, eos_token_id):
        """Mean hidden state of the response tokens, at every layer."""
        attn = sequences.ne(pad_token_id).long().to(sequences.device)
        with torch.no_grad():
            outputs = model(
                input_ids=sequences, attention_mask=attn, output_hidden_states=True, return_dict=True
            )

        gen_ids = sequences[:, start_idx:]
        mask = gen_ids.ne(pad_token_id)
        if eos_token_id is not None:
            mask &= gen_ids.ne(eos_token_id)

        per_layer = [
            masked_mean(h[:, start_idx:, :].float(), mask) for h in outputs.hidden_states
        ]
        del outputs
        return torch.stack(per_layer, dim=1)

    def build_dialog_prompt(templates, round_idx, item_idx, prompt, read_response):
        """Round 0: complete the prompt."""
        base = round0_from_templates(templates, prompt)
        if round_idx == 0:
            return base

        dialog = base + "\n" + read_response(0, item_idx)
        for prev in range(1, round_idx):
            dialog += "\n" + templates[2] + "\n" + read_response(prev, item_idx)
        dialog += "\n" + templates[2]
        return dialog

    def run_variant(variant, input_data, model, tokenizer, args, scorer):
        templates = templates_for(variant)
        out_dir = isc_dir(args.model, variant, args.split)
        os.makedirs(out_dir, exist_ok=True)

        def read_response(r, idx):
            with open(isc_item(args.model, variant, r, idx, args.split), "rb") as f:
                return pickle.load(f)["result"]

        for round_idx in range(args.num_rounds):
            pending_prompts, pending_ids = [], []

            for item_idx, prompt in enumerate(input_data):
                if os.path.exists(isc_item(args.model, variant, round_idx, item_idx, args.split)):
                    continue

                chat = [
                    {"role": "system", "content": system_prompt_for(variant)},
                    {"role": "user", "content": build_dialog_prompt(
                        templates, round_idx, item_idx, prompt, read_response)},
                ]
                pending_prompts.append(
                    tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
                )
                pending_ids.append(item_idx)

                is_last = item_idx == len(input_data) - 1
                if len(pending_prompts) < args.batch_size and not is_last:
                    continue

                batch_prompts, batch_ids = pending_prompts[:], pending_ids[:]
                pending_prompts.clear()
                pending_ids.clear()

                inputs = tokenizer(
                    batch_prompts, truncation=True, padding="longest", return_tensors="pt"
                ).to(model.device)

                with torch.no_grad():
                    sequences = model.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_new_tokens=args.max_new_tokens,
                        min_new_tokens=args.min_new_tokens,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                    )

                input_len = inputs["input_ids"].shape[1]
                texts = tokenizer.batch_decode(sequences[:, input_len:], skip_special_tokens=True)

                hidden = per_layer_response_mean(
                    model, sequences, input_len, tokenizer.pad_token_id, tokenizer.eos_token_id
                )
                scores = scorer.score(texts, prompts=[input_data[i] for i in batch_ids])

                for i, idx in enumerate(batch_ids):
                    record = {
                        "result": texts[i],
                        "prompt": input_data[idx],
                        "variant": variant,
                        "round": round_idx,
                        "avg_output_hidden": hidden[i].cpu().numpy().astype(np.float16),
                    }
                    record.update({key: values[i] for key, values in scores.items()})
                    with open(isc_item(args.model, variant, round_idx, idx, args.split), "wb") as f:
                        pickle.dump(record, f)

                headline = scorer.stats_keys[0]
                print(f"[{variant}] round {round_idx}: {batch_ids[-1] + 1}/{len(input_data)} "
                      f"{headline}={np.mean(scores[headline]):.3f}", flush=True)

            torch.cuda.empty_cache()
            gc.collect()

        write_stats(variant, len(input_data), args, scorer)

    def write_stats(variant, num_data, args, scorer):
        """Per-round mean/var for each of the scorer's scalar keys -- what make_figures plots."""
        stats = {}
        for key in scorer.stats_keys:
            scores = np.full((num_data, args.num_rounds), np.nan)
            for d in range(num_data):
                for r in range(args.num_rounds):
                    path = isc_item(args.model, variant, r, d, args.split)
                    if os.path.exists(path):
                        with open(path, "rb") as f:
                            scores[d, r] = pickle.load(f)[key]
            stats[key] = {
                "per_round": [
                    {
                        "round": r,
                        "mean": float(np.nanmean(scores[:, r])),
                        "var": float(np.nanvar(scores[:, r])),
                        "n": int((~np.isnan(scores[:, r])).sum()),
                    }
                    for r in range(args.num_rounds)
                ],
                "per_item": scores.tolist(),
            }

        out = os.path.join(isc_dir(args.model, variant, args.split), "stats.json")
        with open(out, "w") as f:
            json.dump(stats, f)
        headline = scorer.stats_keys[0]
        means = [f"{s['mean']:.3f}" for s in stats[headline]["per_round"]]
        print(f"[{variant}] {headline} by round: {' -> '.join(means)}")

    def main(args):
        seed_everything(87)

        model, tokenizer = load_model(
            args.model, dtype=torch.float16, device_map="auto"
        )
        print(f"{args.model}: {num_layers(model)} layers, storing ell at all {num_layers(model) + 1}")

        scorers = {}
        for variant in args.variants:
            attribute = attribute_for(variant)
            if attribute not in scorers:
                scorers[attribute] = SCORERS[attribute]()

        eval_items = {}
        for variant in args.variants:
            key = (eval_set_for(variant), eval_data_type(variant))
            if key not in eval_items:
                eval_items[key] = select_eval_items(
                    args.data_dir if key[0] == "rtp" else None,
                    key[1], args.num_data, split=args.split, eval_set=key[0],
                )

        for variant in args.variants:
            key = (eval_set_for(variant), eval_data_type(variant))
            print(f"\n=== {variant} (on {key[1]} prompts from {key[0]}, split={args.split}) ===")
            run_variant(variant, eval_items[key], model, tokenizer, args,
                        scorers[attribute_for(variant)])

    parser = argparse.ArgumentParser()
    add_model(parser)
    add_data_dir(parser)
    parser.add_argument("--variants", nargs="+", default=PUBLISHED_VARIANTS,
                        choices=sorted(VARIANTS))
    parser.add_argument("--split", type=str, default="test", choices=["test", "train"],
                        help="'train' backs ell_bar, which must be estimated off the eval items")
    parser.add_argument("--num_data", type=int, default=500)
    parser.add_argument("--num_rounds", type=int, default=5)
    add_generation(parser, batch_size=5, max_new_tokens=200, min_new_tokens=2)
    main(parser.parse_args(argv))


_COMMANDS = {"check_models": _cli_check_models, "prepare_toxicity": _cli_prepare_toxicity, "prepare_moral": _cli_prepare_moral, "contrastive_pairs": _cli_contrastive_pairs, "steering_vectors": _cli_steering_vectors, "self_correction": _cli_self_correction}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] not in _COMMANDS:
        sys.exit("usage: python -m experiments.pipeline <cmd>  (cmd: " + ", ".join(_COMMANDS) + ")")
    _COMMANDS[argv[0]](argv[1:])


if __name__ == "__main__":
    main()
