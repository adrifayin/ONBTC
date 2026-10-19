"""
Final comparison: six Naive Bayes variants and a linear SVM, each under two protocols.

  paper : alpha = 1 for Naive Bayes and C = 10 for the SVM (Rennie et al.'s settings)
  tuned : alpha (or C) chosen by macro-F1 on a validation set carved from the training data

Settings: 20 Newsgroups with 10 random 80/20 splits (headers kept, the paper's protocol),
Reuters-21578 single-label (ModApte), and 20 Newsgroups chronological with headers removed.

Run from the repository root:
    python -m src.experiments                       # everything (the SVM makes this slow)
    python -m src.experiments --n-splits 2 --out trial.csv   # quick trial
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

from src.data import describe, load_20ng, load_reuters_single_label
from src.twcnb import VARIANTS, RennieTransformer, make_variant

RESULTS = Path("results")
ALPHAS = [1.0, 0.3, 0.1, 0.03, 0.01, 0.003]
C_GRID = [0.1, 1.0, 10.0]


def make_svm(C):
    """The paper's baseline: linear SVM, one-vs-rest, on the same transformed features."""
    return make_pipeline(RennieTransformer(), LinearSVC(C=C, dual=True, max_iter=10000))


def vectorize(train_texts, test_texts):
    vectorizer = CountVectorizer()               # vocabulary from training documents only
    return vectorizer.fit_transform(train_texts), vectorizer.transform(test_texts)


def validation_split(X, y, seed):
    """Carve 20% of the TRAINING data out as a validation set. The test set is never touched."""
    _, counts = np.unique(y, return_counts=True)
    strat = y if counts.min() >= 2 else None     # Reuters has 1-document classes: can't stratify
    return train_test_split(X, y, test_size=0.2, stratify=strat, random_state=seed)


def pick_best(make_model, grid, X_tr, y_tr, seed):
    """Return the grid value with the best macro-F1 on the validation set."""
    X_fit, X_val, y_fit, y_val = validation_split(X_tr, y_tr, seed)
    scores = {}
    for value in grid:
        pred = make_model(value).fit(X_fit, y_fit).predict(X_val)
        scores[value] = f1_score(y_val, pred, average="macro", zero_division=0)
    return max(scores, key=scores.get)


def score(model, X_tr, y_tr, X_te, y_te):
    start = time.perf_counter()
    model.fit(X_tr, y_tr)
    fit_time = time.perf_counter() - start
    pred = model.predict(X_te)
    return {"accuracy": accuracy_score(y_te, pred),
            "macro_f1": f1_score(y_te, pred, average="macro", zero_division=0),
            "fit_time_s": fit_time}


def run_split(X_tr, X_te, y_tr, y_te, setting, split):
    rows = []
    base = {"setting": setting, "split": split}
    for name in VARIANTS:
        rows.append({**base, "protocol": "paper", "model": name, "param": 1.0,
                     **score(make_variant(name, alpha=1.0), X_tr, y_tr, X_te, y_te)})
        a = pick_best(lambda v: make_variant(name, alpha=v), ALPHAS, X_tr, y_tr, split)
        rows.append({**base, "protocol": "tuned", "model": name, "param": a,
                     **score(make_variant(name, alpha=a), X_tr, y_tr, X_te, y_te)})

    paper_svm = score(make_svm(10.0), X_tr, y_tr, X_te, y_te)
    rows.append({**base, "protocol": "paper", "model": "LinearSVC", "param": 10.0, **paper_svm})
    c = pick_best(make_svm, C_GRID, X_tr, y_tr, split)
    tuned_svm = paper_svm if c == 10.0 else score(make_svm(c), X_tr, y_tr, X_te, y_te)
    rows.append({**base, "protocol": "tuned", "model": "LinearSVC", "param": c, **tuned_svm})
    return rows


def summarize(df):
    order = list(VARIANTS) + ["LinearSVC"]
    summary = (df.groupby(["setting", "protocol", "model"])[["accuracy", "macro_f1", "fit_time_s"]]
                 .agg(["mean", "std"]).round(4))
    for setting in df["setting"].unique():
        for protocol in ["paper", "tuned"]:
            print(f"\n=== {setting} | {protocol} ===")
            print(summary.loc[(setting, protocol)].reindex(order).to_string())


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-splits", type=int, default=10, help="random 20 Newsgroups splits")
    parser.add_argument("--settings", nargs="+", choices=["random", "reuters", "bydate"],
                        default=["random", "reuters", "bydate"])
    parser.add_argument("--out", default="final_results.csv", help="file name inside results/")
    args = parser.parse_args()

    rows = []
    if "random" in args.settings:
        for split in range(args.n_splits):
            tr_text, te_text, y_tr, y_te = load_20ng("random", seed=split)
            X_tr, X_te = vectorize(tr_text, te_text)
            start = time.perf_counter()
            rows += run_split(X_tr, X_te, y_tr, y_te, "20NG random", split)
            print(f"20NG random: split {split + 1}/{args.n_splits} done "
                  f"in {time.perf_counter() - start:.0f}s")
    if "reuters" in args.settings:
        tr_text, te_text, y_tr, y_te = load_reuters_single_label()
        describe("Reuters", y_tr, y_te)
        X_tr, X_te = vectorize(tr_text, te_text)
        rows += run_split(X_tr, X_te, y_tr, y_te, "Reuters", 0)
        print("Reuters done")
    if "bydate" in args.settings:
        tr_text, te_text, y_tr, y_te = load_20ng("bydate")
        X_tr, X_te = vectorize(tr_text, te_text)
        rows += run_split(X_tr, X_te, y_tr, y_te, "20NG bydate", 0)
        print("20NG bydate done")

    RESULTS.mkdir(exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / args.out, index=False)
    summarize(df)
    print(f"\nSaved results/{args.out}")


if __name__ == "__main__":
    main()