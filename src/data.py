"""Datasets, the mu-source registry, Moral Stories splits, and the tidy results table."""

import os
import random

import numpy as np
import pandas as pd

from src.paths import ETHICS_DIR, JIGSAW_DIR, MORAL_DIR, RTP_DIR

KEY_COLUMNS = [
    "model", "mu_source", "eval_set", "condition", "variant",
    "data_type", "split",
    "round", "layer", "alpha", "item_idx",
]
METRIC_COLUMNS = [
    "roberta_tox", "detoxify_tox", "ppl", "judge_fluency", "bertscore_f1", "distinct2",
    "cola_fluency",
    "moral_score", "rougeL_moral", "rougeL_immoral", "bleu_moral", "bleu_immoral",
]
COLUMNS = KEY_COLUMNS + ["text"] + METRIC_COLUMNS

CONDITIONS = ("baseline", "isc", "transplant", "steer")
MU_SOURCE_NONE = "-"

_INT_COLUMNS = ["round", "layer", "item_idx"]
_FLOAT_COLUMNS = ["alpha"] + METRIC_COLUMNS


def make_row(model, condition, item_idx, text, mu_source=MU_SOURCE_NONE, eval_set="rtp",
             variant="-", data_type="-", split="test",
             round=0, layer=None, alpha=None, **extra):
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}; expected one of {CONDITIONS}")
    row = {
        "model": model, "mu_source": mu_source, "eval_set": eval_set,
        "condition": condition, "variant": variant,
        "data_type": data_type, "split": split,
        "round": round, "layer": layer,
        "alpha": alpha, "item_idx": item_idx, "text": text,
    }
    row.update({m: np.nan for m in METRIC_COLUMNS})
    row.update(extra)
    return row


def to_frame(rows) -> pd.DataFrame:
    df = pd.DataFrame(list(rows))
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    df = df[COLUMNS]
    for c in _INT_COLUMNS:
        df[c] = df[c].astype("Int64")
    for c in _FLOAT_COLUMNS:
        df[c] = df[c].astype("float64")
    return df


def dedupe(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse rows that describe the same generation."""
    key = df[KEY_COLUMNS].astype(str)
    return df[~key.duplicated()].reset_index(drop=True)


def merge(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Left-join `new`'s metric values onto `old` by key; add rows `old` has never seen."""
    if old is None or old.empty:
        return dedupe(new)

    combined = pd.concat([dedupe(old), dedupe(new)], ignore_index=True)
    key = combined[KEY_COLUMNS].astype(str).agg("|".join, axis=1)
    out = combined.groupby(key, sort=False).agg(
        {c: "last" for c in ["text"]} | {c: _last_valid for c in METRIC_COLUMNS}
        | {c: "first" for c in KEY_COLUMNS}
    )
    return out[COLUMNS].reset_index(drop=True)


def _last_valid(s: pd.Series):
    s = s.dropna()
    return s.iloc[-1] if len(s) else np.nan


def results_path(model_basename: str, results_dir: str = "results") -> str:
    return os.path.join(results_dir, f"{model_basename}.parquet")


def write(df: pd.DataFrame, path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def read(path: str) -> pd.DataFrame:
    """Read, backfilling any column added to the schema since the file was written."""
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = np.nan

    df["tox_mean"] = df[["roberta_tox", "detoxify_tox"]].mean(axis=1, skipna=True)
    for c in _INT_COLUMNS:
        df[c] = df[c].astype("Int64")
    for c in _FLOAT_COLUMNS:
        df[c] = df[c].astype("float64")
    return df[COLUMNS + ["tox_mean"]]


EVAL_SEED = 87

_JIGSAW_LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


EVAL_SETS = {
    "rtp": {
        "dir": RTP_DIR,
        "files": {"test": "{dt}_test_1k.jsonl", "train": "{dt}_train_4k.jsonl"},
        "data_types": ("toxic", "non_toxic"),
    },
    "moral": {
        "dir": MORAL_DIR,
        "files": {"test": "{dt}_eval.jsonl", "train": "{dt}_train.jsonl"},
        "data_types": ("prefix",),
    },
}


def select_eval_items(
    data_dir: str, data_type: str, n: int, split: str = "test", seed: int = EVAL_SEED,
    eval_set: str = "rtp",
):
    """The n prompts every generation stage must agree on."""
    spec = EVAL_SETS[eval_set]
    if data_type not in spec["data_types"]:
        raise ValueError(
            f"eval_set {eval_set!r} has data_types {spec['data_types']}, got {data_type!r}"
        )
    path = os.path.join(data_dir or spec["dir"], spec["files"][split].format(dt=data_type))
    df = pd.read_json(path, lines=True)
    df = df.sample(n=min(n, len(df)), random_state=seed)
    return [row["text"] for row in df["prompt"]]


def load_rtp(data_dir: str = RTP_DIR, limit: int = None):
    """RTP train split."""
    tox = pd.read_json(os.path.join(data_dir, "toxic_train_4k.jsonl"), lines=True)
    non = pd.read_json(os.path.join(data_dir, "non_toxic_train_4k.jsonl"), lines=True)

    t = [(r["text"], r["toxicity"]) for r in tox["prompt"]]
    nt = [(r["text"], r["toxicity"]) for r in non["prompt"]]

    n = min(len(t), len(nt), limit or len(t))
    return t[:n], nt[:n]


def load_jigsaw(data_dir: str = JIGSAW_DIR, limit: int = 4000, min_chars: int = 20,
                length_match: bool = True):
    """Jigsaw 2018 train.csv -> (toxic, non_toxic) pairs, LENGTH-MATCHED."""
    df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    df["comment_text"] = df["comment_text"].astype(str).str.strip()
    df = df[df["comment_text"].str.len() >= min_chars]
    df["severity"] = df[_JIGSAW_LABELS].sum(axis=1) / len(_JIGSAW_LABELS)

    tox_df = df[df["toxic"] == 1].sort_values("severity", ascending=False)
    non_df = df[df["toxic"] == 0]

    n = min(len(tox_df), len(non_df), limit)
    tox_df = tox_df.head(n)

    if length_match:
        non_df = _match_by_length(tox_df["comment_text"], non_df, "comment_text")
    else:
        non_df = non_df.sample(n=n, random_state=EVAL_SEED)

    t = list(zip(tox_df["comment_text"], tox_df["severity"].astype(float)))
    nt = list(zip(non_df["comment_text"], non_df["severity"].astype(float)))
    return t, nt


def _match_by_length(targets, pool: pd.DataFrame, text_col: str) -> pd.DataFrame:
    """For each target text, take the unused pool row closest in character length."""
    target_lens = np.asarray([len(s) for s in targets], dtype=float)
    if len(pool) < len(target_lens):
        raise ValueError(
            f"length-matching pool ({len(pool)}) is smaller than the target set "
            f"({len(target_lens)}); the match would degenerate"
        )

    pool_lens = pool[text_col].str.len().to_numpy(dtype=float)
    taken = []
    for tl in target_lens:
        i = int(np.argmin(np.abs(pool_lens - tl)))
        taken.append(i)
        pool_lens[i] = np.inf
    return pool.iloc[taken]


def load_moral(data_dir: str = MORAL_DIR, limit: int = None):
    """The mu-build pairs for moral acceptability -> (neg, pos) = (immoral, moral)."""
    def read(name):
        df = pd.read_json(os.path.join(data_dir, name), lines=True)
        return [(row["text"], float(row["toxicity"])) for row in df["prompt"]]

    pos = read("moral_mu.jsonl")
    neg = read("immoral_mu.jsonl")

    n = min(len(pos), len(neg), limit or len(pos))
    return neg[:n], pos[:n]


def load_ethics(data_dir: str = ETHICS_DIR, limit: int = None):
    """ETHICS commonsense pairs -> (neg, pos) = (immoral, moral)."""
    return load_moral(data_dir, limit)


MU_SOURCES = {
    "rtp": {
        "loader": load_rtp, "default_dir": RTP_DIR,
        "eval_set": "rtp", "attribute": "toxicity",
        "classes": {"pos": "non_toxic", "neg": "toxic"},
        "high_score_slot": "neg",
        "inject": {"toxic": +1.0, "non_toxic": -1.0},
    },
    "jigsaw": {
        "loader": load_jigsaw, "default_dir": JIGSAW_DIR,
        "eval_set": "rtp", "attribute": "toxicity",
        "classes": {"pos": "non_toxic", "neg": "toxic"},
        "high_score_slot": "neg",
        "inject": {"toxic": +1.0, "non_toxic": -1.0},
    },
    "moral": {
        "loader": load_moral, "default_dir": MORAL_DIR,
        "eval_set": "moral", "attribute": "moral",
        "classes": {"pos": "moral", "neg": "immoral"},
        "high_score_slot": "pos",
        "inject": {"prefix": +1.0},
    },
    "ethics": {
        "loader": load_ethics, "default_dir": ETHICS_DIR,
        "eval_set": "moral", "attribute": "moral",
        "classes": {"pos": "moral", "neg": "immoral"},
        "high_score_slot": "pos",
        "inject": {"prefix": +1.0},
    },
}

def source_for_tag(tag: str) -> str:
    """'steering_moral' -> 'moral'."""
    body = tag[len("steering_"):] if tag.startswith("steering_") else tag
    source = body.split("_", 1)[0].split("-", 1)[0]
    if source not in MU_SOURCES:
        raise ValueError(
            f"steering tag {tag!r} names mu-source {source!r}, which no MU_SOURCES entry "
            f"declares. Known sources: {sorted(MU_SOURCES)}"
        )
    return source


def injection_arms(source: str, alpha_abs: float):
    """The [(data_type, signed_alpha), ...] one |alpha| expands to in the Fig-1 sweep."""
    inject = MU_SOURCES[source]["inject"]
    if len(inject) > 1:
        arms = [(dt, sign * alpha_abs) for dt, sign in inject.items()]
    else:
        (data_type, sign), = inject.items()
        arms = [(data_type, +sign * alpha_abs), (data_type, -sign * alpha_abs)]
    return sorted(arms, key=lambda a: a[1])


SEED = 87


def allocate(all_ids, minimal_ids, lexical_ids, norm_ids, norm_train_ids,
             norm_dev_ids=(), seed: int = SEED, n_mu: int = 512, n_eval: int = 500,
             n_ellbar: int = 200):
    """-> {"mu", "eval", "ellbar", "scorer_train", "scorer_dev"} -> lists of story IDs."""
    rng = random.Random(seed)

    def take(pool, n, what):
        pool = sorted(pool)
        if len(pool) < n:
            raise ValueError(f"{what} needs {n} ids but only {len(pool)} are available")
        rng.shuffle(pool)
        return pool[:n]

    official_test = set(minimal_ids) | set(lexical_ids) | set(norm_ids)

    mu = take(set(minimal_ids), n_mu, "mu")

    free = set(all_ids) - official_test
    eval_ids = take(free, n_eval, "eval")
    ellbar = take(free - set(eval_ids), n_ellbar, "ellbar")

    spent = official_test | set(mu) | set(eval_ids) | set(ellbar)
    scorer_train = sorted(set(norm_train_ids) - spent)
    scorer_dev = sorted(set(norm_dev_ids) - spent - set(scorer_train))

    return {"mu": mu, "eval": eval_ids, "ellbar": ellbar,
            "scorer_train": scorer_train, "scorer_dev": scorer_dev}
