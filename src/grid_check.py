"""
Extended smoothing grid. In the final run, MNB always chose the smallest alpha
tried (0.003) and WCNB always the largest (1), so their best values may lie
outside the grid. This re-tunes every Naive Bayes variant on the original and a
wider grid over the same 10 random 20 Newsgroups splits. No SVM, so it is quick.
Run from the repository root:  python -m src.grid_check
"""
import argparse
from pathlib import Path

import pandas as pd

from src.data import load_20ng
from src.experiments import ALPHAS, pick_best, score, vectorize
from src.twcnb import VARIANTS, make_variant

WIDE_ALPHAS = [10.0, 3.0, 1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001]
RESULTS = Path("results")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-splits", type=int, default=10)
    args = parser.parse_args()

    rows = []
    for split in range(args.n_splits):
        tr_text, te_text, y_tr, y_te = load_20ng("random", seed=split)
        X_tr, X_te = vectorize(tr_text, te_text)
        for name in VARIANTS:
            for grid_name, grid in [("original", ALPHAS), ("wide", WIDE_ALPHAS)]:
                a = pick_best(lambda v: make_variant(name, alpha=v), grid, X_tr, y_tr, split)
                rows.append({"split": split, "model": name, "grid": grid_name, "alpha": a,
                             **score(make_variant(name, alpha=a), X_tr, y_tr, X_te, y_te)})
        print(f"split {split + 1}/{args.n_splits} done")

    RESULTS.mkdir(exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "grid_check.csv", index=False)

    order = list(VARIANTS)
    print("\nMean test accuracy, original vs wide grid:")
    print(df.pivot_table(index="model", columns="grid", values="accuracy", aggfunc="mean")
            .reindex(order).round(4).to_string())
    print("\nAlpha chosen on the wide grid (how often):")
    wide = df[df["grid"] == "wide"]
    print(wide.groupby("model")["alpha"]
              .apply(lambda s: ", ".join(f"{v:g} ({n})" for v, n in s.value_counts().items()))
              .reindex(order).to_string())


if __name__ == "__main__":
    main()