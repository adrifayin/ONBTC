"""
Original experiment: how does each variant cope as the training data gets skewed?

Half of the 20 Newsgroups classes (chosen at random) are shrunk to a fraction of
their training documents. The test set is left untouched, so it stays balanced.

Every variant is run under both protocols, so no model gets an unfair smoothing advantage:
  paper : alpha = 1
  tuned : alpha chosen by 3-fold cross-validation (macro-F1) on the skewed training
          data itself, as a practitioner facing that data would do.
Run from the repository root:  python -m src.imbalance
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, f1_score, recall_score

from src.data import load_20ng
from src.experiments import ALPHAS, pick_best
from src.twcnb import VARIANTS, make_variant

FRACTIONS = [1.0, 0.5, 0.2, 0.1, 0.05]
RESULTS = Path("results")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()

    train_texts, test_texts, y_train_all, y_test = load_20ng("bydate")
    train_texts = np.array(train_texts, dtype=object)
    y_train_all, y_test = np.asarray(y_train_all), np.asarray(y_test)
    classes = np.unique(y_train_all)

    rows = []
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        shrunk = rng.choice(classes, size=len(classes) // 2, replace=False)
        # Shuffle each shrunk class once, so smaller fractions are subsets of larger ones
        order = {c: rng.permutation(np.flatnonzero(y_train_all == c)) for c in shrunk}

        for frac in FRACTIONS:
            keep = np.ones(len(y_train_all), dtype=bool)
            for idx in order.values():
                keep[idx[max(1, int(round(frac * len(idx)))):]] = False

            vectorizer = CountVectorizer()
            X_train = vectorizer.fit_transform(train_texts[keep])
            X_test = vectorizer.transform(test_texts)
            y_train = y_train_all[keep]

            for variant in VARIANTS:
                tuned = pick_best(lambda v: make_variant(variant, alpha=v), ALPHAS,
                                  X_train, y_train, seed, metric="macro_f1")
                for protocol, alpha in [("paper", 1.0), ("tuned", tuned)]:
                    pred = make_variant(variant, alpha=alpha).fit(X_train, y_train).predict(X_test)
                    rows.append({
                        "seed": seed, "fraction": frac, "model": variant,
                        "protocol": protocol, "alpha": alpha,
                        "accuracy": accuracy_score(y_test, pred),
                        "macro_f1": f1_score(y_test, pred, average="macro", zero_division=0),
                        "shrunk_class_recall": recall_score(y_test, pred, labels=shrunk,
                                                            average="macro", zero_division=0),
                    })
            print(f"seed {seed}, fraction {frac:.2f}: done", flush=True)

    RESULTS.mkdir(exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "imbalance.csv", index=False)
    labels = list(VARIANTS)
    for protocol in ["paper", "tuned"]:
        sub = df[df["protocol"] == protocol]
        for metric in ["macro_f1", "shrunk_class_recall"]:
            table = sub.pivot_table(index="fraction", columns="model", values=metric, aggfunc="mean")
            print(f"\n[{protocol}] mean {metric} by fraction of shrunk-class documents kept:")
            print(table.sort_index(ascending=False)[labels].round(4).to_string())
    print("\n[tuned] alpha chosen (most common) by fraction:")
    print(df[df["protocol"] == "tuned"].pivot_table(
        index="fraction", columns="model", values="alpha",
        aggfunc=lambda s: s.mode().iloc[0]).sort_index(ascending=False)[labels].to_string())


if __name__ == "__main__":
    main()
