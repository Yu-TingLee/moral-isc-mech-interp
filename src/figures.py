"""The paper's figures and tables: palette, data loaders, the three compact figures, and the LaTeX tables."""

import functools
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import read
from src.paths import (
    OUTPUTS, alignment_json, isc_dir, mu_injection_root, shift_injection_root,
)

MU   = "#2a78d6"
CTRL = "#eda100"
WEAK = "#b8b6ad"
RED  = "#e34948"
INK  = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e5e4df"

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "grid.color": GRID, "grid.linewidth": 0.6,
})


def model_label(name):
    """Display name for a model: the results/ directory name minus the '-Instruct' tag, e.g. Mistral-7B-Instruct-v0.3 -> Mistral-7B-v0.3."""
    return re.sub(r"-Instruct", "", name)


def outpath(name):
    """Figures and tables go to outputs/."""
    os.makedirs(OUTPUTS, exist_ok=True)
    return os.path.join(OUTPUTS, name)


DEGEN_FRAC = 0.95


def _pick(roberta, detoxify, scorer):
    """One classifier or their mean; toxicity is the mean of the two."""
    if scorer == "roberta" or detoxify is None:
        return roberta
    if scorer == "detoxify":
        return detoxify
    return float(np.mean([roberta, detoxify]))


def _d2(t):
    w = str(t).split()
    bg = list(zip(w, w[1:]))
    return len(set(bg)) / len(bg) if bg else np.nan


def _norm_injection(model, scorer="mean", arm="+"):
    """One signed arm of the mu_rtp sweep."""
    root = mu_injection_root(model, "steering_rtp")
    pattern = "toxic_*" if arm == "+" else "non_toxic_-*"
    rows = []
    for d in glob.glob(os.path.join(root, pattern)):
        base_name = os.path.basename(d)
        if arm == "+" and base_name.startswith("non_toxic"):
            continue
        alpha = float(base_name.rsplit("_", 1)[1])
        if (alpha > 0) != (arm == "+") or alpha == 0:
            continue
        for f in glob.glob(os.path.join(d, "*.json")):
            r = json.load(open(f))
            for k in [k for k in r if k.startswith("steered_L") and k.endswith("_tox")]:
                layer = int(k[len("steered_L"):-len("_tox")])
                det = r.get(f"steered_L{layer}_detoxify")
                rows.append(dict(coef=abs(alpha), layer=layer,
                                 tox=_pick(r[k], det, scorer),
                                 text=r.get(f"steered_L{layer}_text", "")))
    return pd.DataFrame(rows)


@functools.lru_cache(maxsize=None)
def _injection_baseline(model, scorer="mean", arm="+"):
    """Mean unsteered toxicity on the same items the injection sweep runs on."""
    root = mu_injection_root(model, "steering_rtp")
    pattern = "toxic_*" if arm == "+" else "non_toxic_-*"
    vals = []
    for d in glob.glob(os.path.join(root, pattern)):
        base_name = os.path.basename(d)
        if arm == "+" and base_name.startswith("non_toxic"):
            continue
        alpha = float(base_name.rsplit("_", 1)[1])
        if (alpha > 0) != (arm == "+") or alpha == 0:
            continue
        for f in glob.glob(os.path.join(d, "*.json")):
            r = json.load(open(f))
            if "baseline_tox" in r:
                vals.append(_pick(r["baseline_tox"], r.get("baseline_detoxify"), scorer))
    return float(np.mean(vals)) if vals else None


def _transplant(model, scorer="mean", arm="+"):
    """ell_bar transplanted into a plain round-0 turn."""
    root = shift_injection_root(model, "steering_rtp")
    pattern = "toxic_a*n_L*" if arm == "+" else "non_toxic_a*n_L*"
    rows = []
    for d in glob.glob(os.path.join(root, pattern)):
        if arm == "+" and os.path.basename(d).startswith("non_toxic"):
            continue
        b = os.path.basename(d)
        alpha = float(b.split("_a")[1].split("n_")[0]); layer = int(b.split("_L")[1])
        for f in glob.glob(os.path.join(d, "*.json")):
            r = json.load(open(f))
            rows.append(dict(coef=alpha, layer=layer,
                             tox=_pick(r["transplant_tox"],
                                       r.get("transplant_detoxify"), scorer),
                             text=r["transplant_text"]))
    return pd.DataFrame(rows)


def _moral_injection(model, arm):
    """One signed arm of the mu_moral sweep."""
    root = mu_injection_root(model, "steering_moral")
    rows = []
    for d in glob.glob(os.path.join(root, "prefix_*")):
        alpha = float(os.path.basename(d).rsplit("_", 1)[1])
        if (alpha > 0) != (arm == "+") or alpha == 0:
            continue
        for f in glob.glob(os.path.join(d, "*.json")):
            r = json.load(open(f))
            for k in [k for k in r if k.startswith("steered_L") and k.endswith("_moral")]:
                layer = int(k[len("steered_L"):-len("_moral")])
                rows.append(dict(coef=abs(alpha), layer=layer, score=r[k],
                                 base=r.get("baseline_moral"),
                                 text=r.get(f"steered_L{layer}_text", ""),
                                 base_text=r.get("baseline_text", "")))
    return pd.DataFrame(rows)


def _moral_transplant(model, arm="+"):
    """ell_bar_moral transplanted into a plain round-0 turn."""
    root = shift_injection_root(model, "steering_moral")
    pattern = "prefix_a*n_L*" if arm == "+" else "prefix_immoralize_strong_a*n_L*"
    rows = []
    for d in glob.glob(os.path.join(root, pattern)):
        if arm == "+" and "immoralize" in os.path.basename(d):
            continue
        b = os.path.basename(d)
        alpha = float(b.split("_a")[1].split("n_")[0]); layer = int(b.split("_L")[1])
        for f in glob.glob(os.path.join(d, "*.json")):
            r = json.load(open(f))
            if "transplant_moral" not in r:
                continue
            rows.append(dict(coef=alpha, layer=layer, score=r["transplant_moral"],
                             text=r.get("transplant_text", "")))
    return pd.DataFrame(rows)


DEPTH = np.linspace(0.0, 1.0, 41)

DASH = (0, (5, 2))
NULL_DASH = (0, (6, 3))
NULL_DOT = (0, (1, 2))

ARMS = {
    "toxicity": ("steering_rtp", [
        ("strong_detox", MU, "-", "Strong-Detox"),
        ("weak_detox", MU, DASH, "Weak-Detox"),
        ("strong_tox", RED, "-", "Strong-Tox"),
        ("weak_tox", RED, DASH, "Weak-Tox"),
        ("neutral_concise", CTRL, "-", "Neutral control"),
    ], "Toxicity"),
    "moral": ("steering_moral", [
        ("moralize_strong", MU, "-", "Strong-Moralize"),
        ("moralize_weak", MU, DASH, "Weak-Moralize"),
        ("immoralize_strong", RED, "-", "Strong-Immoralize"),
        ("immoralize_weak", RED, DASH, "Weak-Immoralize"),
        ("moral_neutral", CTRL, "-", "Neutral control"),
    ], "Morality"),
}


def _out(name, out_dir):
    """outputs/ by default; the paper's newfigs/ when a caller names one."""
    if out_dir is None:
        return outpath(name)
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, name)


def _legend(fig, handles, ncol, y):
    fig.legend(handles, [h.get_label() for h in handles], loc="lower center", ncol=ncol,
               frameon=False, bbox_to_anchor=(0.5, y))


ROW_IN = 0.23


def _legend_rows(fig, rows, y, dy=None):
    """One `fig.legend` per row, each independently centred; `rows[0]` is the top row."""
    dy = ROW_IN / fig.get_figheight() if dy is None else dy
    for i, row in enumerate(reversed(rows)):
        fig.legend(row, [h.get_label() for h in row], loc="lower center", ncol=len(row),
                   frameon=False, bbox_to_anchor=(0.5, y + i * dy))


def _suffix(mode):
    """'final' (the published cos(ell^(L), mu^(l))) keeps the plain filename; anything else is suffixed so it cannot overwrite it."""
    return "" if mode == "final" else f"_{mode}"


def _centre(curves, center):
    """The bold across-model curve; 'median' is the published (unsuffixed) choice."""
    if center not in ("median", "mean"):
        raise ValueError(f"center must be 'median' or 'mean', not {center!r}")
    return np.mean(curves, axis=0) if center == "mean" else np.median(curves, axis=0)


def _csuffix(center):
    return "_mean" if center == "mean" else ""


def _have(models, tags, mode):
    """Models with an alignment json for `mode` under any of `tags`; guards against an empty figure when nothing has been measured yet."""
    return [m for m in models
            if any(os.path.exists(alignment_json(m, t, mode)) for t in tags)]


def _curves(models, tag, variant, mode="final"):
    """One curve per model, resampled onto relative depth."""
    out = []
    for m in models:
        path = alignment_json(m, tag, mode)
        if not os.path.exists(path):
            continue
        d = pd.DataFrame(json.load(open(path)))
        d = d[(d["round"] == 1) & (d.variant == variant)].sort_values("layer")
        if d.empty:
            continue
        x = d.layer.to_numpy(dtype=float)
        out.append(np.interp(DEPTH, x / x.max() if x.max() else x, d.cos_mu_bar.to_numpy()))
    return np.asarray(out)


def _null_level(models, tag, key, mode="final"):
    vals = []
    for m in models:
        path = alignment_json(m, tag, mode)
        if not os.path.exists(path):
            continue
        d = pd.DataFrame(json.load(open(path)))
        d = d[d["round"] == 1]
        if key in d:
            vals.append(float(d[key].mean()))
    return float(np.mean(vals)) if vals else None


def _base_d2(model, arm):
    """Mean distinct-2 of the unsteered generations -- the degeneracy gate's denominator."""
    root = mu_injection_root(model, "steering_rtp")
    pattern = "toxic_*" if arm == "+" else "non_toxic_-*"
    vals = []
    for d in glob.glob(os.path.join(root, pattern)):
        if arm == "+" and os.path.basename(d).startswith("non_toxic"):
            continue
        for f in glob.glob(os.path.join(d, "*.json")):
            r = json.load(open(f))
            if "baseline_text" in r:
                vals.append(_d2(r["baseline_text"]))
    return float(np.nanmean(vals)) if vals else np.nan


def _best(frame, col, base_d2, better, coef, gate=True):
    """Steered score at one coefficient, at the layer with the largest effect among layers surviving the degeneracy gate."""
    if frame is None or frame.empty:
        return np.nan
    g = frame[frame.coef == coef].groupby("layer").agg(
        s=(col, "mean"),
        d2=("text", lambda z: float(np.nanmean([_d2(x) for x in z]))),
    ).reset_index()
    if gate and np.isfinite(base_d2):
        g = g[g.d2 >= DEGEN_FRAC * base_d2]
    if g.empty:
        return np.nan
    return float(g.s.max() if better == "max" else g.s.min())


COEFS = (0.5, 1.0, 2.0)
COEF_MARK = {0.5: "^", 1.0: "o", 2.0: "s"}


def _extreme(frame, col, base_d2, better, gate=True):
    """(score, coef) for the coefficient that pushes the metric furthest, over the layers `_best` admits."""
    best = (np.nan, None)
    for c in COEFS:
        v = _best(frame, col, base_d2, better, c, gate=gate)
        if v is None or np.isnan(v):
            continue
        if np.isnan(best[0]) or (v > best[0] if better == "max" else v < best[0]):
            best = (v, c)
    return best


def _toxicity_effects(model, gate=True):
    """-> {direction: (baseline, (mu, coef), (ell, coef))}, absolute toxicity."""
    out = {}
    for direction, sign in (("benign", "+"), ("harmful", "-")):
        base, bd2 = _injection_baseline(model, arm=sign), _base_d2(model, sign)
        if base is None:
            continue
        better = "min" if sign == "+" else "max"
        mu_f, ell_f = _norm_injection(model, arm=sign), _transplant(model, arm=sign)
        out[direction] = (float(base),
                          _extreme(mu_f, "tox", bd2, better, gate=gate),
                          _extreme(ell_f, "tox", bd2, better, gate=gate))
    return out


def _moral_effects(model, gate=True):
    """-> {direction: (baseline, (mu, coef), (ell, coef))}, absolute p(normative)."""
    pos = _moral_injection(model, "+")
    if pos.empty:
        return {}
    base = float(pos.base.dropna().mean())
    bd2 = float(np.nanmean([_d2(t) for t in pos.base_text.dropna()]))
    out = {}
    for direction, arm, better in (("benign", "+", "max"), ("harmful", "-", "min")):
        mu_f = pos if arm == "+" else _moral_injection(model, "-")
        ell_f = _moral_transplant(model, arm=arm)
        out[direction] = (base,
                          _extreme(mu_f, "score", bd2, better, gate=gate),
                          _extreme(ell_f, "score", bd2, better, gate=gate))
    return out


MS = 5.2

PROMPT_VARIANT = {
    ("toxicity", "benign"): "strong_detox", ("toxicity", "harmful"): "strong_tox",
    ("moral", "benign"): "moralize_strong", ("moral", "harmful"): "immoralize_strong",
}


def _prompt_r1(model, attr, direction):
    """Round-1 score of the strong prompt, from the same two sources the tables read -- the parquet for toxicity, stats.json for moral."""
    variant = PROMPT_VARIANT[(attr, direction)]
    try:
        if attr == "moral":
            path = os.path.join(isc_dir(model, variant), "stats.json")
            per_round = json.load(open(path))["moral_score"]["per_round"]
            return float(per_round[1]["mean"])
        df = read(f"results/{model}.parquet")
        g = df[(df.condition == "isc") & (df.split == "test")
               & (df.variant == variant) & (df["round"] == 1)]
        return float(g.tox_mean.mean()) if not g.empty else np.nan
    except Exception:
        return np.nan


INJ_PANELS = [
    ("toxicity", "benign", _toxicity_effects, "Detoxify"),
    ("toxicity", "harmful", _toxicity_effects, "Toxify"),
    ("moral", "benign", _moral_effects, "Moralize"),
    ("moral", "harmful", _moral_effects, "Immoralize"),
]
XLABEL = {"toxicity": "toxicity", "moral": "moral score"}


def _inj_suffix(with_prompt, gate):
    """The published form keeps the plain filename; a deviation is named for how it deviates."""
    return ("" if with_prompt else "_noprompt") + ("_gated" if gate else "")


def fig_injection_compact(models, out_dir=None, with_prompt=True, gate=False):
    """Attribute score after injection, on the metric's own 0-1 scale, per coefficient."""
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.0), sharex="row", squeeze=False)
    ys = np.arange(len(models))[::-1]

    for k, (attr, direction, getter, title) in enumerate(INJ_PANELS):
        ax = axes[k // 2][k % 2]
        for y, model in zip(ys, models):
            try:
                eff = getter(model, gate=gate)
            except Exception as e:
                print(f"  SKIP {attr}/{direction} {model}: {type(e).__name__}: {e}")
                continue
            if direction not in eff:
                continue
            base, (mu_v, mu_c), (ell_v, ell_c) = eff[direction]
            prompt = _prompt_r1(model, attr, direction) if with_prompt else np.nan
            pts = [base] + [v for v in (mu_v, ell_v, prompt)
                            if v is not None and not np.isnan(v)]
            if len(pts) > 1:
                ax.plot([min(pts), max(pts)], [y, y], "-", color=WEAK, lw=1.3, zorder=2)
            ax.plot([base], [y], marker="o", ms=MS, color=WEAK, mec=INK2, mew=0.9,
                    ls="none", zorder=3)
            if not np.isnan(prompt):
                ax.plot([prompt], [y], marker="|", ms=MS * 1.9, mew=1.3, color=INK2,
                        ls="none", zorder=3.5)
            for colour, v, c in ((MU, mu_v, mu_c), (RED, ell_v, ell_c)):
                if v is not None and not np.isnan(v) and c is not None:
                    ax.plot([v], [y], marker=COEF_MARK[c], ms=MS, color=colour,
                            ls="none", zorder=4)

        ax.set_yticks(ys)
        ax.set_yticklabels([model_label(m) for m in models] if k % 2 == 0 else [], fontsize=7.2)
        ax.set_title(title, fontsize=9)
        ax.set_xlim(-0.06, 1.06)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.grid(axis="x", lw=0.6, alpha=0.8)
        ax.set_axisbelow(True)
        ax.set_ylim(-0.8, len(models) - 0.2)
        ax.set_xlabel(XLABEL[attr], fontsize=8.5)

    handles = [plt.Line2D([], [], color=WEAK, mec=INK2, mew=0.9, marker="o", ms=MS,
                          ls="none", label="no injection")]
    if with_prompt:
        handles.append(plt.Line2D([], [], color=INK2, marker="|", ms=MS * 1.9, mew=1.3,
                                  ls="none", label="strong prompting (R1)"))
    handles += [plt.Line2D([], [], color=MU, marker="o", ms=MS, ls="none",
                           label=r"steer with $\mu$"),
                plt.Line2D([], [], color=RED, marker="o", ms=MS, ls="none",
                           label=r"steer with $\ell$")]
    handles += [plt.Line2D([], [], color=INK2, marker=COEF_MARK[c], ms=MS, ls="none",
                           label=rf"$|\alpha|={c:g}$") for c in COEFS]
    if with_prompt:
        _legend_rows(fig, [handles[:4], handles[4:]], -0.10)
    else:
        _legend(fig, handles, 6, -0.07)
    fig.tight_layout()
    out = _out(f"injection_compact{_inj_suffix(with_prompt, gate)}.png", out_dir)
    fig.savefig(out, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


TRANSFER = {
    "toxicity": ("steering_rtp", "steering_jigsaw"),
    "moral": ("steering_moral", "steering_ethics"),
}

ELL_TRANSFER_TITLE = {
    "toxicity": r"Toxicity: RTP $\rightarrow$ Jigsaw",
    "moral": r"Morality: Moral Stories $\rightarrow$ ETHICS",
}


def _depth_panel(ax, models, tag, attr, title, mode="final", center="median"):
    """One panel of the depth-pooled alignment form: cos(ell_1, mu^(l)) against relative depth, one thin line per model per arm, the across-model centre in bold, and a +-1 std band."""
    for variant, colour, ls, _label in ARMS[attr][1]:
        curves = _curves(models, tag, variant, mode)
        if not len(curves):
            continue
        for c in curves:
            ax.plot(DEPTH, c, "-", color=colour, lw=0.8, alpha=0.22, zorder=2)
        mid = _centre(curves, center)
        sd = np.std(curves, axis=0)
        ax.fill_between(DEPTH, mid - sd, mid + sd, color=colour, alpha=0.12, lw=0, zorder=2)
        ax.plot(DEPTH, mid, ls=ls, color=colour, lw=1.5, zorder=4)

    rnull = _null_level(models, tag, "cos_random_bar", mode)
    if rnull is not None:
        ax.axhline(rnull, color=WEAK, ls=NULL_DASH, lw=1.2, zorder=3)
    gnull = _null_level(models, tag, "cos_gauss_bar", mode)
    if gnull is not None:
        ax.axhline(gnull, color=WEAK, ls=NULL_DOT, lw=1.4, zorder=3)

    ax.axhline(0, color=INK2, lw=0.8, zorder=1)
    ax.set_xlim(0, 1)
    ax.set_xlabel("relative depth")
    ax.set_title(title, fontsize=9)
    ax.grid(axis="y", lw=0.6, alpha=0.8)
    ax.set_axisbelow(True)


IN_CORPUS_TITLE = {
    "toxicity": "Toxicity: $\\mu$ from RTP",
    "moral": "Morality: $\\mu$ from Moral Stories",
}


def fig_alignment_depth_compact(models, out_dir=None, mode="final", center="median"):
    """In-corpus alignment: the model axis collapsed onto relative depth, drawn exactly the way `ell_transfer_compact.png` is, so the two can be set side by side."""
    if not _have(models, [tag for tag, _, _ in ARMS.values()], mode):
        print(f"  SKIP fig_alignment_depth_compact(mode={mode!r}): nothing measured yet")
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.9), sharey=True)

    for ax, (attr, (tag, _arms, _title)) in zip(axes, ARMS.items()):
        _depth_panel(ax, models, tag, attr, IN_CORPUS_TITLE[attr], mode, center)

    handles = [
        plt.Line2D([], [], color=MU, ls="-", lw=2.0, label="Strong Detox/Moralize"),
        plt.Line2D([], [], color=MU, ls=DASH, lw=2.0, label="Weak Detox/Moralize"),
        plt.Line2D([], [], color=RED, ls="-", lw=2.0, label="Strong Tox/Immoralize"),
        plt.Line2D([], [], color=RED, ls=DASH, lw=2.0, label="Weak Tox/Immoralize"),
        plt.Line2D([], [], color=CTRL, ls="-", lw=2.0, label="Neutral control"),
        plt.Line2D([], [], color=WEAK, ls=NULL_DASH, lw=1.2, label=r"shuffled $\mu$"),
        plt.Line2D([], [], color=WEAK, ls=NULL_DOT, lw=1.4, label="random direction"),
    ]
    _legend_rows(fig, [handles[:4], handles[4:]], -0.17)
    fig.tight_layout()
    out = _out(f"alignment_depth_compact{_suffix(mode)}{_csuffix(center)}.png", out_dir)
    fig.savefig(out, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def fig_ell_transfer_compact(models, out_dir=None, mode="final", center="median"):
    """Does ell still find mu once mu is rebuilt from the out-of-corpus source? One thin line per (model, variant), the across-model centre, and a +-1 std band per variant."""
    if not _have(models, [tag_b for _, tag_b in TRANSFER.values()], mode):
        print(f"  SKIP fig_ell_transfer_compact(mode={mode!r}): nothing measured yet")
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.9), sharey=True)

    for ax, (attr, (_tag_a, tag_b)) in zip(axes, TRANSFER.items()):
        _depth_panel(ax, models, tag_b, attr, ELL_TRANSFER_TITLE[attr], mode, center)

    handles = [
        plt.Line2D([], [], color=MU, ls="-", lw=2.0, label="Strong Detox/Moralize"),
        plt.Line2D([], [], color=MU, ls=DASH, lw=2.0, label="Weak Detox/Moralize"),
        plt.Line2D([], [], color=RED, ls="-", lw=2.0, label="Strong Tox/Immoralize"),
        plt.Line2D([], [], color=RED, ls=DASH, lw=2.0, label="Weak Tox/Immoralize"),
        plt.Line2D([], [], color=CTRL, ls="-", lw=2.0, label="Neutral control"),
        plt.Line2D([], [], color=WEAK, ls=NULL_DASH, lw=1.2, label=r"shuffled $\mu$"),
        plt.Line2D([], [], color=WEAK, ls=NULL_DOT, lw=1.4, label="random direction"),
    ]
    _legend_rows(fig, [handles[:4], handles[4:]], -0.17)
    fig.tight_layout()
    out = _out(f"ell_transfer_compact{_suffix(mode)}{_csuffix(center)}.png", out_dir)
    fig.savefig(out, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


MODEL_ORDER = [
    "LFM2-2.6B-Exp", "Mistral-7B-Instruct-v0.3", "Qwen2.5-3B-Instruct",
    "Qwen2.5-7B-Instruct", "zephyr-7b-alpha", "zephyr-7b-beta",
]


def _display(model):
    return model.replace("-Instruct", "")


def _ordered(models):
    """Models present, in the paper's order; then any extras, so a new model still prints."""
    present = set(models)
    return [m for m in MODEL_ORDER if m in present] + [m for m in models if m not in MODEL_ORDER]


def _pair(strong, weak, fmt="{:.3f}"):
    """One cell: strong and weak arms side by side, an em dash for an absent arm."""
    fs = "\\textemdash" if strong is None else fmt.format(strong)
    fw = "\\textemdash" if weak is None else fmt.format(weak)
    return f"{fs} $\\mid$ {fw}"


def _delta_pair(strong, weak):
    """Signed strong and weak cell for the Delta_1 table; the larger magnitude is bolded."""
    def one(v, bold):
        sign = "+" if v >= 0 else "-"
        body = f"{abs(v):.3f}"
        return f"${sign}\\textbf{{{body}}}$" if bold else f"${sign}{body}$"
    strong_wins = abs(strong) >= abs(weak)
    return f"{one(strong, strong_wins)} $\\mid$ {one(weak, not strong_wins)}"


def _tox_frame(model):
    """The isc/test rows of one model's parquet, or None if it has not been ingested."""
    df = read(f"results/{model}.parquet")
    if df is None:
        return None
    return df[(df.condition == "isc") & (df.split == "test")]


def _tox_cell(frame, variant, round_, col, min_words=0):
    g = frame[(frame.variant == variant) & (frame["round"] == round_)]
    if min_words:
        g = g[g.text.map(lambda t: len(str(t).split()) >= min_words)]
    return float(g[col].mean()) if not g.empty else None


def _moral_rounds(model, variant):
    """[R0, R1, ...] moral scores from run_self_correction's stats.json, or None if absent."""
    path = os.path.join(isc_dir(model, variant), "stats.json")
    if not os.path.exists(path):
        return None
    return [r["mean"] for r in json.load(open(path))["moral_score"]["per_round"]]


_TOX_METRICS = [
    ("Toxicity", "tox_mean", "{:.3f}", 0),
    (r"PPL $\downarrow$", "ppl", "{:.1f}", 5),
    (r"Fluency $\uparrow$", "judge_fluency", "{:.2f}", 0),
]
_TOX_ROUNDS = [0, 1, 2, 3, 4]


def render_toxicity_rounds(models):
    body = [
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\caption{\textbf{Toxicity evaluation across models}. Results are formatted as "
        r"strong $\mid$ weak prompting.}",
        r"\label{tab:toxicity_rounds}",
        r"\resizebox{0.94\textwidth}{!}{%",
        r"\begin{tabular}{llcccccccccc}",
        r"\toprule",
        r"\multirow{2}{*}{\textbf{Model}} & \multirow{2}{*}{\textbf{Metric}} & "
        r"\multicolumn{5}{c}{\textbf{Detoxify}} & \multicolumn{5}{c}{\textbf{Toxify}} \\",
        r"\cmidrule(lr){3-7} \cmidrule(lr){8-12}",
        r" & & \textbf{R0} & \textbf{R1} & \textbf{R2} & \textbf{R3} & \textbf{R4} & "
        r"\textbf{R0} & \textbf{R1} & \textbf{R2} & \textbf{R3} & \textbf{R4} \\",
        r"\midrule",
    ]
    for i, model in enumerate(_ordered(models)):
        frame = _tox_frame(model)
        if frame is None or frame.empty:
            continue
        if i:
            body.append(r"\midrule")
        body.append(rf"\multirow{{3}}{{*}}{{{_display(model)}}} ")
        for label, col, fmt, min_words in _TOX_METRICS:
            cells = []
            for arm_strong, arm_weak in (("strong_detox", "weak_detox"),
                                         ("strong_tox", "weak_tox")):
                for r in _TOX_ROUNDS:
                    s = _tox_cell(frame, arm_strong, r, col, min_words)
                    w = _tox_cell(frame, arm_weak, r, col, min_words)
                    cells.append(_pair(s, w, fmt))
            body.append(f" & {label} & " + " & ".join(cells) + r" \\")
    body += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}"]
    return "\n".join(body)


def render_moral_correction(models):
    body = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{\textbf{Morality evaluation across models.} Metic is moral score. "
        r"Results are formatted as strong $\mid$ weak prompting.}",
        r"\label{tab:moral_correction}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\multirow{2}{*}{\textbf{Model}} & \multicolumn{2}{c}{\textbf{Moralize}} & "
        r"\multicolumn{2}{c}{\textbf{Immoralize}} \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}",
        r" & \textbf{R0} & \textbf{R1} & \textbf{R0} & \textbf{R1} \\",
        r"\midrule",
    ]
    for model in _ordered(models):
        ms_s, ms_w = _moral_rounds(model, "moralize_strong"), _moral_rounds(model, "moralize_weak")
        im_s, im_w = _moral_rounds(model, "immoralize_strong"), _moral_rounds(model, "immoralize_weak")
        if not (ms_s and im_s):
            continue
        cells = [
            _pair(ms_s[0], ms_w[0] if ms_w else None),
            _pair(ms_s[1], ms_w[1] if ms_w else None),
            _pair(im_s[0], im_w[0] if im_w else None),
            _pair(im_s[1], im_w[1] if im_w else None),
        ]
        body.append(f"{_display(model)} & " + " & ".join(cells) + r" \\")
    body += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}"]
    return "\n".join(body)


def render_round1_change(models):
    body = [
        r"\begin{table}[th!]",
        r"\centering",
        r"\caption{\textbf{Round 1 change.} $\Delta_1 =$ \textbf{R1} $-$ \textbf{R0}, "
        r"formatted as strong $\mid$ weak prompts. Bold marks the larger absolute value.}",
        r"\label{tab:round1_change}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\multirow{2}{*}{\textbf{Model}} & \multicolumn{2}{c}{\textbf{Toxicity} $\Delta_1$} "
        r"& \multicolumn{2}{c}{\textbf{Moral Score} $\Delta_1$} \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}",
        r" & \textbf{Detoxify} $\downarrow$ & \textbf{Toxify} $\uparrow$ & "
        r"\textbf{Moralize} $\uparrow$ & \textbf{Immoralize} $\downarrow$ \\",
        r"\midrule",
    ]
    for model in _ordered(models):
        frame = _tox_frame(model)
        ms_s, ms_w = _moral_rounds(model, "moralize_strong"), _moral_rounds(model, "moralize_weak")
        im_s, im_w = _moral_rounds(model, "immoralize_strong"), _moral_rounds(model, "immoralize_weak")
        if frame is None or frame.empty or not (ms_s and ms_w and im_s and im_w):
            continue

        def tox_delta(variant):
            return _tox_cell(frame, variant, 1, "tox_mean") - _tox_cell(frame, variant, 0, "tox_mean")

        cells = [
            _delta_pair(tox_delta("strong_detox"), tox_delta("weak_detox")),
            _delta_pair(tox_delta("strong_tox"), tox_delta("weak_tox")),
            _delta_pair(ms_s[1] - ms_s[0], ms_w[1] - ms_w[0]),
            _delta_pair(im_s[1] - im_s[0], im_w[1] - im_w[0]),
        ]
        body.append(f"{_display(model)} & " + " & ".join(cells) + r" \\")
    body += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}"]
    return "\n".join(body)


def write_tables(models):
    """Write the three result tables to outputs/tables.tex, in the paper's order."""
    sections = [
        render_toxicity_rounds(models),
        render_moral_correction(models),
        render_round1_change(models),
    ]
    header = ("% Experimental-results tables for "
              "Explaining Intrinsic Moral Self-Correction with Mechanistic Interpretability.\n"
              "% Generated by `python -m experiments.make_figures` from results/.\n"
              "% Formatting matches paper-draft/main.tex exactly.\n")
    path = outpath("tables.tex")
    open(path, "w").write(header + "\n" + "\n\n".join(sections) + "\n")
    print("wrote", path)
