"""Every output path in one place."""

import os

RESULTS = "results"
OUTPUTS = "outputs"

DATA = "data"
RTP_DIR = os.path.join(DATA, "rtp")
JIGSAW_DIR = os.path.join(DATA, "jigsaw")
PROCESSED_DIR = os.path.join(DATA, "processed")
MORAL_DIR = os.path.join(DATA, "moral")
ETHICS_DIR = os.path.join(DATA, "ethics")

SCORER_DIR = os.path.join(RESULTS, "_scorers")


def processed_dir(source, processed_root: str = PROCESSED_DIR) -> str:
    """data/processed/{source}/ -- build_contrastive_pairs writes it; build_steering_vectors and measure_residual_norms's probe read it."""
    return os.path.join(processed_root, str(source))


PAIR_SLOTS = ("pos", "neg")

_LEGACY_SLOT_FILES = {"pos": "nt_prompt_score.jsonl", "neg": "t_prompt_score.jsonl"}


def pair_file(source, slot: str, processed_root: str = PROCESSED_DIR) -> str:
    """data/processed/{source}/{slot}_prompt_score.jsonl, falling back to the legacy name."""
    if slot not in PAIR_SLOTS:
        raise ValueError(f"unknown slot {slot!r}; expected one of {PAIR_SLOTS}")
    directory = processed_dir(source, processed_root)
    new = os.path.join(directory, f"{slot}_prompt_score.jsonl")
    if os.path.exists(new):
        return new
    legacy = os.path.join(directory, _LEGACY_SLOT_FILES[slot])
    return legacy if os.path.exists(legacy) else new


def classes_json(source, processed_root: str = PROCESSED_DIR) -> str:
    """The manifest recording which class is in which slot, and mu's sign convention."""
    return os.path.join(processed_dir(source, processed_root), "classes.json")


def basename(model_name: str) -> str:
    return os.path.basename(model_name.rstrip("/"))


def model_dir(model_name: str) -> str:
    return os.path.join(RESULTS, basename(model_name))


def mu_transfer_json(model_name: str, attribute: str = "toxicity") -> str:
    """cos(mu_A^(l), mu_B^(l)) per layer for one attribute -- measure_mu_transfer.py writes it, the transfer figure reads it."""
    leaf = "mu_transfer.json" if attribute == "toxicity" else f"mu_transfer_{attribute}.json"
    return os.path.join(model_dir(model_name), leaf)


def steering_tag(sources, data_ratio: float = 1.0, suffix: str = "") -> str:
    """The name of a mu and everything derived from it: `steering_{source}`, e.g. sources=['rtp'] -> 'steering_rtp'."""
    if isinstance(sources, str):
        sources = [sources]
    tag = "steering_" + "-".join(str(s) for s in sources)
    if data_ratio != 1.0:
        tag += f"_t{int(data_ratio * 100)}"
    return f"{tag}_{suffix}" if suffix else tag


def isc_dir(model_name: str, variant: str, split: str = "test") -> str:
    """ell_bar is estimated on the *train* split, which must not overwrite the eval runs -- hence the suffix."""
    leaf = variant if split == "test" else f"{variant}_{split}"
    return os.path.join(model_dir(model_name), "isc", leaf)


def isc_item(model_name: str, variant: str, round_idx: int, item_idx: int,
             split: str = "test") -> str:
    return os.path.join(isc_dir(model_name, variant, split), f"{round_idx}_{item_idx:05d}.pkl")


def steering_dir(model_name: str, tag: str) -> str:
    return os.path.join(model_dir(model_name), tag)


def vector_dir(model_name: str, tag: str) -> str:
    return os.path.join(steering_dir(model_name, tag), "steering_vectors")


def steer_json(model_name: str, tag: str) -> str:
    return os.path.join(vector_dir(model_name, tag), "steer.json")


def mu_injection_dir(model_name: str, tag: str, data_type: str, alpha: float) -> str:
    """Inject alpha * mu_hat^(l), where alpha is a *fraction of the residual norm* at the hook site (alpha * median||x^(l)||), not an absolute coefficient."""
    return os.path.join(
        steering_dir(model_name, tag), "mu_injection", "sequential", f"{data_type}_{alpha}"
    )


def residual_norms_dir(model_name: str, attribute: str = "toxicity") -> str:
    """Per-layer ||x^(l)||, plus mu/ell_bar norms and a linear-probe check."""
    leaf = "residual_norms" if attribute == "toxicity" else f"residual_norms_{attribute}"
    return os.path.join(model_dir(model_name), leaf)


def alignment_dir(model_name: str, tag: str) -> str:
    return os.path.join(steering_dir(model_name, tag), "alignment")


def alignment_json(model_name: str, tag: str, mode: str = "final") -> str:
    """`mode` is measure_alignment's --mode, and the two are NOT interchangeable: final cos(ell^(L), mu^(l)) -- ell read at the last layer and held fixed while mu varies."""
    return os.path.join(alignment_dir(model_name, tag), f"cossim_{mode}.json")


def mu_injection_root(model_name: str, tag: str) -> str:
    return os.path.join(steering_dir(model_name, tag), "mu_injection", "sequential")


def shift_injection_root(model_name: str, tag: str) -> str:
    return os.path.join(steering_dir(model_name, tag), "shift_injection")


def shift_injection_dir(model_name: str, tag: str, data_type: str, alpha: float, layer: int,
                        alpha_mode: str = "norm", arm: str = None) -> str:
    """Inject alpha * ell_bar_hat^(l) -- the prompt's OWN mean displacement, estimated on the train split -- into a generation with no revise instruction in context."""
    token = f"a{alpha}n" if alpha_mode == "norm" else f"a{alpha}"
    stem = data_type if arm is None else f"{data_type}_{arm}"
    return os.path.join(
        steering_dir(model_name, tag), "shift_injection", f"{stem}_{token}_L{layer}"
    )
