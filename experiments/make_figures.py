"""Build the paper's figures and tables into outputs/."""

import argparse
import glob
import os

from experiments._args import add_models
from src.figures import (
    fig_alignment_depth_compact, fig_ell_transfer_compact, fig_injection_compact, write_tables,
)
from src.paths import OUTPUTS, RESULTS


def build(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except Exception as e:
        detail = ", ".join(f"{k}={v!r}" for k, v in kwargs.items())
        print(f"  SKIP {fn.__name__}({detail}): {type(e).__name__}: {e}")


def main(models):
    print("models:", models)
    build(fig_alignment_depth_compact, models)
    build(fig_ell_transfer_compact, models)
    build(fig_injection_compact, models)
    build(write_tables, models)
    print("\nEverything ->", OUTPUTS + "/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_models(ap)
    a = ap.parse_args()
    main(a.models or [os.path.basename(p)[:-8]
                      for p in sorted(glob.glob(os.path.join(RESULTS, "*.parquet")))])
