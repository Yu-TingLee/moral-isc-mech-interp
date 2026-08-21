"""Argparse fragments shared by the experiment entry points."""

import argparse

from src.paths import RTP_DIR

DEFAULT_STEERING_TAG = "steering_rtp"


def add_model(p: argparse.ArgumentParser, required: bool = True) -> None:
    """The model to work on: a Hugging Face id, or the basename of one."""
    p.add_argument("--model", required=required, metavar="HF_ID",
                   help="e.g. Qwen/Qwen2.5-3B-Instruct")


def add_models(p: argparse.ArgumentParser) -> None:
    """Several models at once, for stages that build cross-model figures."""
    p.add_argument("--models", nargs="+", default=None, metavar="HF_ID",
                   help="default: every model with a results/*.parquet")


def add_steering_tag(p: argparse.ArgumentParser, default: str = DEFAULT_STEERING_TAG) -> None:
    """Which mu to use, named for its source corpus: steering_rtp, steering_jigsaw, steering_moral, steering_ethics."""
    p.add_argument("--steering_tag", default=default)


def add_limit(p: argparse.ArgumentParser, default: int = None) -> None:
    """How many eval items."""
    p.add_argument("--limit", type=int, default=default)


def add_data_dir(p: argparse.ArgumentParser, default: str = RTP_DIR) -> None:
    p.add_argument("--data_dir", default=default)


def add_generation(p: argparse.ArgumentParser, batch_size: int = 8,
                   max_new_tokens: int = 200, min_new_tokens: int = 0) -> None:
    """Generation budget."""
    p.add_argument("--batch_size", type=int, default=batch_size)
    p.add_argument("--max_new_tokens", type=int, default=max_new_tokens)
    p.add_argument("--min_new_tokens", type=int, default=min_new_tokens)


def add_injection(p: argparse.ArgumentParser) -> None:
    """Where the vector is added, shared by the two injection sweeps."""
    p.add_argument("--layer_stride", type=int, default=1,
                   help="1 = every layer, which is what the figures draw")
    p.add_argument("--inject_offset", type=int, default=0)
    p.add_argument("--apply_during_decode", action="store_true",
                   help="default is prefill only; decode-time steering is a different "
                        "intervention and is not what the paper reports")
