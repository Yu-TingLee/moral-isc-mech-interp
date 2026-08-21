"""Residual norms, ell-mu alignment, and mu cross-corpus transfer."""

import sys
import argparse
import json
import os
import pickle
import numpy as np
import pandas as pd
import torch
from experiments._args import add_model, add_steering_tag
from src.data import EVAL_SETS, select_eval_items
from src.model import PROMPT_PREFIX, chat_wrap
from src.model import load_model, num_layers
from src.paths import PROCESSED_DIR, residual_norms_dir, isc_dir, isc_item, pair_file, steer_json
from src.model import masked_mean
from src.utils import seed_everything
from src.variants import attribute_for, eval_set_for, round0_prompt, system_prompt_for
import matplotlib.pyplot as plt
from experiments._args import add_limit, add_model, add_steering_tag
from src.paths import alignment_dir, isc_item, steer_json
from src.variants import VARIANTS
from experiments._args import add_model
from src.data import source_for_tag
from src.paths import OUTPUTS, basename, mu_transfer_json, steer_json


def _cli_residual_norms(argv):
    DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}

    ALPHAS_SWEPT = [1.0, 2.0, 3.0, 4.0]

    def residual_norms(model, tokenizer, texts, batch_size, max_len):
        """Per-token L2 norm of the residual stream, per layer: mean, std, median, p10, p90."""
        device = next(model.parameters()).device
        n_states = num_layers(model) + 1
        s1 = torch.zeros(n_states, dtype=torch.float64)
        s2 = torch.zeros(n_states, dtype=torch.float64)
        samples = [[] for _ in range(n_states)]
        count = 0.0

        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            enc = tokenizer(
                batch, padding=True, truncation=True, max_length=max_len, return_tensors="pt"
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            keep = enc["attention_mask"].bool()

            with torch.no_grad():
                out = model(**enc, use_cache=False, output_hidden_states=True)

            m = enc["attention_mask"].to(torch.float64)
            for i, h in enumerate(out.hidden_states):
                norms = h.float().norm(dim=-1).to(torch.float64)
                s1[i] += float((norms * m).sum())
                s2[i] += float((norms.pow(2) * m).sum())
                samples[i].append(norms[keep].cpu().numpy())
            count += float(m.sum())
            del out
            print(f"  norms: {min(start + batch_size, len(texts))}/{len(texts)}", flush=True)

        mean = (s1 / count).numpy()
        std = np.sqrt(np.maximum((s2 / count).numpy() - mean**2, 0.0))
        flat = [np.concatenate(s) for s in samples]
        return {
            "mean": mean,
            "std": std,
            "median": np.array([float(np.median(s)) for s in flat]),
            "p10": np.array([float(np.percentile(s, 10)) for s in flat]),
            "p90": np.array([float(np.percentile(s, 90)) for s in flat]),
            "sink_frac": np.array([float((s > 3 * np.median(s)).mean()) for s in flat]),
        }

    def pooled_states(model, tokenizer, texts, batch_size, max_len):
        """masked_mean of every layer's output over real tokens."""
        device = next(model.parameters()).device
        chunks = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            enc = tokenizer(
                batch, padding=True, truncation=True, max_length=max_len, return_tensors="pt"
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                out = model(**enc, use_cache=False, output_hidden_states=True)
            pooled = torch.stack(
                [masked_mean(h.float(), enc["attention_mask"]) for h in out.hidden_states], dim=1
            )
            chunks.append(pooled.cpu())
            del out
            print(f"  probe states: {min(start + batch_size, len(texts))}/{len(texts)}", flush=True)
        return torch.cat(chunks).numpy()

    def probe_f1_per_layer(X_tox, X_non, seed=0, test_size=0.25):
        """Logistic probe, toxic (1) vs non-toxic (0), per layer."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import f1_score, roc_auc_score
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        n_states = X_tox.shape[1]
        y = np.concatenate([np.ones(len(X_tox)), np.zeros(len(X_non))])
        idx_tr, idx_te = train_test_split(
            np.arange(len(y)), test_size=test_size, random_state=seed, stratify=y
        )

        rows = []
        for l in range(n_states):
            X = np.concatenate([X_tox[:, l, :], X_non[:, l, :]]).astype(np.float64)
            X = np.nan_to_num(X)
            clf = make_pipeline(
                StandardScaler(), LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
            )
            clf.fit(X[idx_tr], y[idx_tr])
            pred = clf.predict(X[idx_te])
            prob = clf.predict_proba(X[idx_te])[:, 1]
            rows.append(
                {
                    "state": l,
                    "probe_f1": float(f1_score(y[idx_te], pred)),
                    "probe_acc": float((pred == y[idx_te]).mean()),
                    "probe_auc": float(roc_auc_score(y[idx_te], prob)),
                }
            )
            print(f"  probe layer {l}: F1={rows[-1]['probe_f1']:.3f}", flush=True)
        return pd.DataFrame(rows)

    def mu_norms(model_name, tag, n_layers):
        path = steer_json(model_name, tag)
        if not os.path.exists(path):
            print(f"  (no steer.json at {path}; mu columns will be NaN)")
            return np.full(n_layers, np.nan)
        raw = json.load(open(path))
        out = np.full(n_layers, np.nan)
        for k, v in raw.items():
            v = np.asarray(v, dtype=np.float64)
            if v.ndim == 2:
                v = v.mean(axis=0)
            out[int(k)] = float(np.linalg.norm(v))
        return out

    def ell_bar_norms(model_name, variant, n_train, n_layers):
        """||ell_bar^(l)|| = || mean_i (h_1^(l) - h_0^(l)) || over the train ISC run."""
        d = isc_dir(model_name, variant, "train")
        if not os.path.isdir(d):
            print(f"  (no train ISC run at {d}; ell_bar columns will be NaN)")
            return np.full(n_layers, np.nan), np.full(n_layers, np.nan), 0

        shifts = []
        for i in range(n_train):
            try:
                with open(isc_item(model_name, variant, 0, i, "train"), "rb") as f:
                    h0 = pickle.load(f)["avg_output_hidden"]
                with open(isc_item(model_name, variant, 1, i, "train"), "rb") as f:
                    h1 = pickle.load(f)["avg_output_hidden"]
            except FileNotFoundError:
                break
            shifts.append(np.asarray(h1, np.float64) - np.asarray(h0, np.float64))
        if not shifts:
            return np.full(n_layers, np.nan), np.full(n_layers, np.nan), 0

        S = np.stack(shifts)
        mean_shift = S.mean(axis=0)
        bar = np.linalg.norm(mean_shift, axis=-1)
        per_item = np.linalg.norm(S, axis=-1).mean(axis=0)
        return bar[1 : n_layers + 1], per_item[1 : n_layers + 1], len(shifts)

    def plot(df, model_basename, out_path):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
        L = df["layer"]

        ax = axes[0]
        ax.plot(L, df["norm_in"], label=r"median $\|x^{(l)}\|$ (injection site)")
        ax.fill_between(L, df["norm_in_p10"], df["norm_in_p90"], alpha=0.15)
        ax.plot(L, df["norm_in_mean"], lw=1, alpha=0.6,
                label=r"mean $\|x^{(l)}\|$ (sink-inflated)")
        ax.plot(L, df["mu_norm"], label=r"$\|\mu^{(l)}\|$")
        ax.plot(L, df["ell_bar_norm"], label=r"$\|\bar{\ell}^{(l)}\|$")
        for a in ALPHAS_SWEPT[-1:]:
            ax.axhline(a, ls=":", c="k", lw=1)
            ax.text(0.5, a, f"alpha={a:g}", fontsize=8, va="bottom")
        ax.set_yscale("log")
        ax.set_xlabel("layer")
        ax.set_ylabel("L2 norm (log)")
        ax.set_title("What we inject vs. what we inject into")
        ax.legend(fontsize=8)

        ax = axes[1]
        for a in ALPHAS_SWEPT:
            ax.plot(L, a / df["norm_in"], label=f"alpha={a:g}")
        ax.plot(L, df["mu_ratio"], "k--", label=r"$\|\mu\|/\|x\|$ (natural)")
        ax.plot(L, df["ell_ratio"], "k:", label=r"$\|\bar{\ell}\|/\|x\|$ (natural)")
        ax.set_yscale("log")
        ax.set_xlabel("layer")
        ax.set_ylabel(r"effective strength  coef / $\|x^{(l)}\|$")
        ax.set_title("Effective injection strength")
        ax.legend(fontsize=8)

        ax = axes[2]
        ax.plot(L, df["probe_f1"], label="F1")
        ax.plot(L, df["probe_auc"], label="AUC", alpha=0.6)
        ax.axhline(0.5, ls=":", c="k", lw=1)
        ax.set_ylim(0.3, 1.02)
        ax.set_xlabel("layer")
        ax.set_ylabel("held-out score")
        ax.set_title("Linear toxicity probe (train prompts)")
        ax.legend(fontsize=8)

        fig.suptitle(model_basename)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        print(f"wrote {out_path}")

    def main(args):
        seed_everything(0, benchmark=True)
        attribute = attribute_for(args.isc_variant)
        eval_set = eval_set_for(args.isc_variant)
        out_dir = residual_norms_dir(args.model, attribute)
        os.makedirs(out_dir, exist_ok=True)

        model, tokenizer = load_model(
            args.model, dtype=DTYPES[args.dtype], device_map="auto"
        )
        n_layers = num_layers(model)
        print(f"{n_layers} layers, d_model={model.config.hidden_size}")

        eval_prompts = []
        for dt in EVAL_SETS[eval_set]["data_types"]:
            eval_prompts += select_eval_items(args.data_dir, dt, args.n_eval, eval_set=eval_set)

        if attribute == "toxicity":
            turns, system = [PROMPT_PREFIX + p for p in eval_prompts], None
        else:
            turns = [round0_prompt(args.isc_variant, p) for p in eval_prompts]
            system = system_prompt_for(args.isc_variant)
        chats = chat_wrap(tokenizer, turns, system)
        print(f"residual norms over {len(chats)} chat-wrapped eval prompts "
              f"(attribute={attribute}, eval_set={eval_set})")
        N = residual_norms(model, tokenizer, chats, args.batch_size, args.max_len)

        probe = None
        if args.n_probe > 0:
            t = pd.read_json(
                pair_file(args.source, "neg", args.data_processed_dir), lines=True
            )["prompt"].tolist()[: args.n_probe]
            nt = pd.read_json(
                pair_file(args.source, "pos", args.data_processed_dir), lines=True
            )["prompt"].tolist()[: args.n_probe]
            print(f"probe: {len(t)} toxic / {len(nt)} non-toxic raw train prompts")
            X_t = pooled_states(model, tokenizer, t, args.batch_size, args.max_len)
            X_nt = pooled_states(model, tokenizer, nt, args.batch_size, args.max_len)
            probe = probe_f1_per_layer(X_t, X_nt, seed=args.seed, test_size=args.probe_test_size)

        del model
        torch.cuda.empty_cache()

        mu_n = mu_norms(args.model, args.steering_tag, n_layers)
        ell_n, ell_item_n, n_ell = ell_bar_norms(
            args.model, args.isc_variant, args.n_train, n_layers
        )

        rows = []
        for l in range(n_layers):
            row = {
                "model": os.path.basename(args.model),
                "layer": l,
                "norm_in": float(N["median"][l]),
                "norm_in_mean": float(N["mean"][l]),
                "norm_in_std": float(N["std"][l]),
                "norm_in_p10": float(N["p10"][l]),
                "norm_in_p90": float(N["p90"][l]),
                "sink_frac_in": float(N["sink_frac"][l]),
                "norm_out": float(N["median"][l + 1]),
                "norm_out_mean": float(N["mean"][l + 1]),
                "norm_out_std": float(N["std"][l + 1]),
                "mu_norm": float(mu_n[l]),
                "ell_bar_norm": float(ell_n[l]),
                "ell_item_norm": float(ell_item_n[l]),
                "mu_ratio": float(mu_n[l] / N["median"][l + 1]),
                "ell_ratio": float(ell_n[l] / N["median"][l + 1]),
                "alpha_matching_mu": float(mu_n[l]),
                "alpha_matching_ell": float(ell_n[l]),
            }
            for a in ALPHAS_SWEPT:
                row[f"eff_alpha_{a:g}"] = float(a / N["median"][l])
            if probe is not None:
                p = probe.iloc[l + 1]
                row.update(probe_f1=float(p["probe_f1"]), probe_acc=float(p["probe_acc"]),
                           probe_auc=float(p["probe_auc"]))
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_parquet(os.path.join(out_dir, "diagnostics.parquet"), index=False)
        meta = {
            "model": args.model,
            "n_layers": n_layers,
            "dtype": args.dtype,
            "n_eval_prompts": len(chats),
            "n_probe_per_class": args.n_probe,
            "n_ell_train_items": n_ell,
            "steering_tag": args.steering_tag,
            "isc_variant": args.isc_variant,
            "norm_site": "hidden_states[l] = input of block l = forward_pre_hook site (inject_offset=0)",
            "norm_stat": "norm_in/norm_out are the MEDIAN per-token norm; *_mean columns are the "
                         "mean, which massive-activation sink tokens inflate by up to ~2.8x",
            "alphas_swept": ALPHAS_SWEPT,
        }
        with open(os.path.join(out_dir, "diagnostics.json"), "w") as f:
            json.dump({"meta": meta, "layers": rows}, f, indent=2)
        plot(df, os.path.basename(args.model), os.path.join(out_dir, "diagnostics.png"))

        cols = ["layer", "norm_in", "norm_out", "mu_norm", "ell_bar_norm", "mu_ratio", "ell_ratio",
                "eff_alpha_3"] + (["probe_f1"] if probe is not None else [])
        pd.set_option("display.width", 200, "display.max_columns", 20)
        print("\n" + df[cols].iloc[:: max(1, n_layers // 12)].round(4).to_string(index=False))
        print(f"\nwrote {out_dir}/diagnostics.{{json,parquet,png}}")

    p = argparse.ArgumentParser()
    add_model(p)
    p.add_argument("--data_dir", default=None,
                   help="defaults to the eval set registered for --isc_variant")
    p.add_argument("--data_processed_dir", default=PROCESSED_DIR)
    p.add_argument("--source", default="rtp",
                   help="mu-source the probe uses: rtp, jigsaw, moral or ethics. "
                        "Must match --steering_tag.")
    add_steering_tag(p)
    p.add_argument("--isc_variant", default="strong_detox",
                   help="selects the ATTRIBUTE, its eval set and the prefill the residual "
                        "norms are measured under; also the ISC run backing ||ell_bar||. "
                        "moralize_strong for the moral arm.")
    p.add_argument("--n_eval", type=int, default=100, help="eval prompts per split for the norms")
    p.add_argument("--n_probe", type=int, default=1000, help="train prompts per class; 0 to skip")
    p.add_argument("--n_train", type=int, default=200, help="train ISC items backing ell_bar")
    p.add_argument("--probe_test_size", type=float, default=0.25)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_len", type=int, default=128)
    p.add_argument("--dtype", default="float16", choices=sorted(DTYPES))
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args(argv))


def _cli_alignment(argv):
    def load_hidden(model_name, variant, num_data, num_rounds):
        """(num_data, num_rounds, L+1, d) -- the per-layer ell that run_self_correction now stores."""
        items = []
        for d in range(num_data):
            rounds = []
            for r in range(num_rounds):
                path = isc_item(model_name, variant, r, d)
                if not os.path.exists(path):
                    raise FileNotFoundError(f"{path} -- run experiments.pipeline self_correction for variant {variant!r}")
                with open(path, "rb") as f:
                    rounds.append(pickle.load(f)["avg_output_hidden"])
            items.append(np.stack(rounds))
        return np.stack(items).astype(np.float32)

    def load_mu(model_name, tag):
        raw = json.load(open(steer_json(model_name, tag)))
        mu = {}
        for k, v in raw.items():
            v = np.asarray(v, dtype=np.float32)
            if v.ndim == 2:
                v = v.mean(axis=0)
            mu[int(k)] = v
        return mu

    def cos(a, b):
        """Row-wise cosine of (N, d) against a single (d,)."""
        a_n = np.linalg.norm(a, axis=-1) + 1e-12
        return (a @ b) / (a_n * (np.linalg.norm(b) + 1e-12))

    def compute(args):
        mu = load_mu(args.model, args.steering_tag)
        layers = sorted(mu)

        n_null = args.n_null
        rng = np.random.RandomState(0)

        def unit_normals(d, k):
            """k directions drawn uniformly on the unit sphere S^(d-1)."""
            v = rng.normal(size=(k, d))
            return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)

        shuffled = {l: np.stack([rng.permutation(mu[l]) for _ in range(n_null)]) for l in layers}
        gauss = {l: unit_normals(mu[l].shape[0], n_null).astype(mu[l].dtype) for l in layers}

        rows = []
        for variant in args.variants:
            hidden = load_hidden(args.model, variant, args.limit, args.num_rounds)
            shift = hidden[:, 1:] - hidden[:, :-1]

            for r in range(shift.shape[1]):
                for l in layers:
                    ell = shift[:, r, -1, :] if args.mode == "final" else shift[:, r, l + 1, :]
                    ellbar = ell.mean(axis=0)
                    per_item = cos(ell, mu[l])
                    rand_draws = np.array([cos(ell, v).mean() for v in shuffled[l]])
                    gauss_draws = np.array([cos(ell, g).mean() for g in gauss[l]])
                    enorm = np.linalg.norm(shift[:, r, l + 1, :], axis=-1)
                    rows.append({
                        "variant": variant,
                        "round": r + 1,
                        "layer": l,
                        "cos_mu_bar": float(cos(ellbar[None, :], mu[l])[0]),
                        "cos_mu": float(per_item.mean()),
                        "cos_mu_std": float(per_item.std()),
                        "n": int(per_item.shape[0]),
                        "cos_random_bar": float(cos(shuffled[l], ellbar).mean()),
                        "cos_gauss_bar": float(cos(gauss[l], ellbar).mean()),
                        "cos_random": float(np.mean(rand_draws)),
                        "cos_random_sd": float(np.std(rand_draws)),
                        "cos_gauss": float(np.mean(gauss_draws)),
                        "cos_gauss_sd": float(np.std(gauss_draws)),
                        "cos_gauss_absmean": float(np.mean(np.abs(gauss_draws))),
                        "n_null": int(n_null),
                        "ell_norm": float(enorm.mean()),
                        "ell_norm_std": float(enorm.std()),
                    })
        return rows

    def plot(rows, args):
        out_dir = alignment_dir(args.model, args.steering_tag)
        os.makedirs(out_dir, exist_ok=True)
        basename = os.path.basename(args.model)

        with open(os.path.join(out_dir, f"cossim_{args.mode}.json"), "w") as f:
            json.dump(rows, f, indent=2)

        fig, (ax_cos, ax_norm) = plt.subplots(1, 2, figsize=(7.6, 3.4), dpi=300)

        for variant in args.variants:
            for r in range(1, args.num_rounds):
                sel = [x for x in rows if x["variant"] == variant and x["round"] == r]
                if not sel:
                    continue
                xs = [x["layer"] + 1 for x in sel]
                ax_cos.plot(xs, [x["cos_mu_bar"] for x in sel], label=f"{variant} r{r}", linewidth=1.4)
                ax_norm.plot(xs, [x["ell_norm"] for x in sel], label=f"{variant} r{r}", linewidth=1.4)

        sel = [x for x in rows if x["variant"] == args.variants[0] and x["round"] == 1]
        ax_cos.plot([x["layer"] + 1 for x in sel], [x["cos_random_bar"] for x in sel],
                    color="gray", linestyle="--", label="shuffled null", linewidth=1.2)
        ax_cos.plot([x["layer"] + 1 for x in sel], [x["cos_gauss_bar"] for x in sel],
                    color="gray", linestyle=":", label="gaussian null", linewidth=1.2)
        ax_cos.axhline(0, color="black", linewidth=0.8)

        ax_cos.set_xlabel("layer of $\\mu$")
        ax_cos.set_ylabel(r"$\cos(\bar{\ell}_k, \mu^{(l)})$")
        ax_cos.set_title(f"alignment ({args.mode})", fontsize=10)
        ax_cos.legend(fontsize=6, loc="best")

        ax_norm.set_xlabel("layer")
        ax_norm.set_ylabel(r"$\|\ell_k^{(l)}\|$")
        ax_norm.set_title("shift magnitude", fontsize=10)
        ax_norm.legend(fontsize=6, loc="best")

        fig.suptitle(basename, fontsize=12, weight="bold")
        fig.tight_layout()
        out = os.path.join(out_dir, f"{basename}_cossim_{args.mode}.png")
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out}")

        peak = max(rows, key=lambda x: x["cos_mu_bar"])
        print(f"peak alignment: {peak['cos_mu_bar']:.4f} at layer {peak['layer']} "
              f"({peak['variant']}, round {peak['round']}); null ~ {peak['cos_random_bar']:+.4f}")

    p = argparse.ArgumentParser()
    add_model(p)
    add_steering_tag(p)
    p.add_argument("--variants", nargs="+", default=["strong_detox"], choices=sorted(VARIANTS))
    p.add_argument("--mode", default="final", choices=["final", "perlayer"],
                   help="'final' reproduces the published figure -- ell read at the LAST "
                        "layer and held fixed while mu varies; 'perlayer' reads ell at mu's "
                        "own layer, cos(ell^(l), mu^(l))")
    p.add_argument("--n_null", type=int, default=100,
                   help="independent draws per null, for BOTH the shuffled and the gaussian "
                        "one. A single draw is badly biased: on Qwen2.5-3B one permutation "
                        "gives a null of +0.0231 where 16 averaged give -0.0018, so the "
                        "single-draw null was over 10x its own true magnitude.")
    add_limit(p, 500)
    p.add_argument("--num_rounds", type=int, default=5)
    args = p.parse_args(argv)
    plot(compute(args), args)


def _cli_mu_transfer(argv):
    TRANSFER_PAIRS = {
        "toxicity": ("steering_rtp", "steering_jigsaw"),
        "moral": ("steering_moral", "steering_ethics"),
    }

    def load_mu(model_name, tag):
        path = steer_json(model_name, tag)
        if not os.path.exists(path):
            sys.exit(
                f"missing {path}\nBuild it first:\n"
                f"  python -m experiments.pipeline steering_vectors --model {model_name} "
                f"--sources {source_for_tag(tag)}"
            )
        raw = json.load(open(path))
        return {int(l): np.asarray(v, dtype=np.float64) for l, v in raw.items()}

    def cos(a, b):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        return float(a @ b / (na * nb)) if na and nb else 0.0

    def main(args):
        tag_a, tag_b = TRANSFER_PAIRS[args.attribute]
        a_mu = load_mu(args.model, tag_a)
        b_mu = load_mu(args.model, tag_b)

        layers = sorted(set(a_mu) & set(b_mu))
        d = len(a_mu[layers[0]])
        rng = np.random.default_rng(0)

        rows = []
        for l in layers:
            a, b = a_mu[l], b_mu[l]
            rows.append({
                "layer": l,
                "cos": cos(a, b),
                "cos_shuffled": cos(a, rng.permutation(b)),
                "cos_random": cos(a, rng.standard_normal(d)),
                "norm_a": float(np.linalg.norm(a)),
                "norm_b": float(np.linalg.norm(b)),
            })

        real = np.array([r["cos"] for r in rows])
        null = np.array([abs(r["cos_shuffled"]) for r in rows])
        rnd = np.array([abs(r["cos_random"]) for r in rows])

        print(f"{basename(args.model)}   d={d}   {len(layers)} layers")
        print(f"  1/sqrt(d) = {1/np.sqrt(d):.4f}  <- the scale of a chance cosine in this space\n")
        print(f"{'layer':>5} {'cos(A,B)':>16} {'shuffled':>10} {'random':>9}"
              f" {'||mu_A||':>11} {'||mu_B||':>11}")
        for r in rows:
            if r["layer"] % args.print_stride and r["layer"] != layers[-1]:
                continue
            print(f"{r['layer']:>5} {r['cos']:>16.4f} {r['cos_shuffled']:>10.4f}"
                  f" {r['cos_random']:>9.4f} {r['norm_a']:>11.2f} {r['norm_b']:>11.2f}")

        print(f"\n  mean cos  = {real.mean():+.4f}   (min {real.min():+.4f}, max {real.max():+.4f})")
        print(f"  mean null = {null.mean():.4f} shuffled / {rnd.mean():.4f} random")
        print(f"  signal / shuffled-null = {abs(real.mean()) / max(null.mean(), 1e-9):.1f}x")
        print(f"  layers with cos > 0 : {(real > 0).sum()}/{len(real)}")

        out = mu_transfer_json(args.model, args.attribute)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            json.dump({"model": basename(args.model), "d_model": d, "layers": rows}, f, indent=2)
        print(f"\nwrote {out}")

        if args.plot:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            ls = [r["layer"] for r in rows]
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(ls, real, marker="o", ms=3, label=r"$\cos(\mu_A^{(l)},\ \mu_B^{(l)})$")
            ax.plot(ls, [r["cos_shuffled"] for r in rows], ls="--", lw=1, c="gray",
                    label="shuffled null")
            ax.axhline(0, c="k", lw=0.5)
            ax.axhspan(-3/np.sqrt(d), 3/np.sqrt(d), color="gray", alpha=0.15,
                       label=r"$\pm 3/\sqrt{d}$ (chance)")
            ax.set_xlabel("layer"); ax.set_ylabel("cosine similarity")
            ax.set_title(f"Does the {args.attribute} direction survive a change of corpus?\n"
                         f"{basename(args.model)}")
            ax.legend(fontsize=8); ax.set_ylim(-0.2, 1.0)
            fig.tight_layout()
            os.makedirs(OUTPUTS, exist_ok=True)
            p = os.path.join(OUTPUTS, f"mu_transfer_{args.attribute}_{basename(args.model)}.png")
            fig.savefig(p, dpi=150)
            print(f"wrote {p}")

    p = argparse.ArgumentParser()
    add_model(p)
    p.add_argument("--attribute", default="toxicity", choices=sorted(TRANSFER_PAIRS))
    p.add_argument("--print_stride", type=int, default=3)
    p.add_argument("--plot", action="store_true", default=True)
    main(p.parse_args(argv))


_COMMANDS = {"residual_norms": _cli_residual_norms, "alignment": _cli_alignment, "mu_transfer": _cli_mu_transfer}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] not in _COMMANDS:
        sys.exit("usage: python -m experiments.measure <cmd>  (cmd: " + ", ".join(_COMMANDS) + ")")
    _COMMANDS[argv[0]](argv[1:])


if __name__ == "__main__":
    main()
