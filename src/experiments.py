"""
Final comparison: six Naive Bayes variants and a linear SVM, each under two protocols.

  paper : alpha = 1 for Naive Bayes and C = 10 for the SVM (Rennie et al.'s settings)
  tuned : alpha (or C) chosen by 3-fold stratified cross-validation on the TRAINING data,
          scored with the metric the setting reports (accuracy for 20 Newsgroups,
          macro-F1 for Reuters). The test set is never used for tuning.

Settings: 20 Newsgroups with 10 random 80/20 splits (headers kept, the paper's protocol),
Reuters-21578 single-label (ModApte), and 20 Newsgroups chronological with headers removed.
Reuters and the chronological split have one fixed test set each, so their per-document
predictions are saved to results/predictions/ for bootstrap confidence intervals
(python -m src.significance).

Run from the repository root:
    python -m src.experiments                       # everything (the SVM makes this slow)
    python -m src.experiments --n-splits 2 --out trial.csv   # quick trial
"""
import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

from src.data import describe, load_20ng, load_reuters_single_label
from src.twcnb import VARIANTS, RennieTransformer, make_variant

RESULTS = Path("results")
PREDICTIONS = RESULTS / "predictions"
ALPHAS = [1.0, 0.3, 0.1, 0.03, 0.01, 0.003]
C_GRID = [0.1, 1.0, 10.0]
CV_FOLDS = 3
N_JOBS = 6
# Tune on the metric each setting reports, so the selection and the evaluation agree.
TUNE_METRIC = {"20NG random": "accuracy", "Reuters": "macro_f1", "20NG bydate": "accuracy"}


def make_svm(C):
    """The paper's baseline: linear SVM, one-vs-rest, on the same transformed features.

    scikit-learn's default squared hinge loss; the paper's SvmFu used the standard hinge.
    src.fidelity_checks compares the two (same accuracy, the hinge is slower to train).
    """
    return make_pipeline(RennieTransformer(), LinearSVC(C=C, dual=True, max_iter=10000))


def vectorize(train_texts, test_texts):
    vectorizer = CountVectorizer()               # vocabulary from training documents only
    return vectorizer.fit_transform(train_texts), vectorizer.transform(test_texts)


def metric_score(metric, y_true, y_pred):
    if metric == "accuracy":
        return accuracy_score(y_true, y_pred)
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def cv_folds(y, seed):
    """Stratified folds over the TRAINING data. Reuters has classes with fewer documents
    than folds; scikit-learn then warns and spreads them as well as it can."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
        return list(skf.split(np.zeros(len(y)), y))


def _fold_score(model, X, y, train_idx, val_idx, metric):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")          # SVM convergence notes on small folds
        pred = model.fit(X[train_idx], y[train_idx]).predict(X[val_idx])
    return metric_score(metric, y[val_idx], pred)


def pick_best(make_model, grid, X_tr, y_tr, seed, metric="macro_f1", n_jobs=N_JOBS):
    """Return the grid value with the best mean cross-validated score (ties: earlier value)."""
    y_tr = np.asarray(y_tr)
    folds = cv_folds(y_tr, seed)
    jobs = [(v, tr, va) for v in grid for tr, va in folds]
    scores = Parallel(n_jobs=n_jobs)(
        delayed(_fold_score)(make_model(v), X_tr, y_tr, tr, va, metric) for v, tr, va in jobs)
    mean = {v: np.mean(scores[i * len(folds):(i + 1) * len(folds)]) for i, v in enumerate(grid)}
    return max(grid, key=lambda v: mean[v])


def score(model, X_tr, y_tr, X_te, y_te):
    start = time.perf_counter()
    model.fit(X_tr, y_tr)
    fit_time = time.perf_counter() - start
    pred = model.predict(X_te)
    return {"accuracy": accuracy_score(y_te, pred),
            "macro_f1": f1_score(y_te, pred, average="macro", zero_division=0),
            "fit_time_s": fit_time}, pred


def run_split(X_tr, X_te, y_tr, y_te, setting, split):
    """Returns (result rows, {"protocol|model": test predictions})."""
    rows, preds = [], {}
    base = {"setting": setting, "split": split}
    metric = TUNE_METRIC[setting]

    def record(protocol, model, param, result):
        stats, pred = result
        rows.append({**base, "protocol": protocol, "model": model, "param": param, **stats})
        preds[f"{protocol}|{model}"] = pred

    for name in VARIANTS:
        record("paper", name, 1.0, score(make_variant(name, alpha=1.0), X_tr, y_tr, X_te, y_te))
        a = pick_best(lambda v: make_variant(name, alpha=v), ALPHAS, X_tr, y_tr, split, metric)
        record("tuned", name, a, score(make_variant(name, alpha=a), X_tr, y_tr, X_te, y_te))

    paper_svm = score(make_svm(10.0), X_tr, y_tr, X_te, y_te)
    record("paper", "LinearSVC", 10.0, paper_svm)
    c = pick_best(make_svm, C_GRID, X_tr, y_tr, split, metric)
    record("tuned", "LinearSVC", c,
           paper_svm if c == 10.0 else score(make_svm(c), X_tr, y_tr, X_te, y_te))
    return rows, preds


def save_predictions(setting, y_true, preds):
    PREDICTIONS.mkdir(parents=True, exist_ok=True)
    path = PREDICTIONS / f"{setting.replace(' ', '_').lower()}.npz"
    np.savez_compressed(path, y_true=np.asarray(y_true), **preds)


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
            rows += run_split(X_tr, X_te, y_tr, y_te, "20NG random", split)[0]
            print(f"20NG random: split {split + 1}/{args.n_splits} done "
                  f"in {time.perf_counter() - start:.0f}s", flush=True)
    if "reuters" in args.settings:
        tr_text, te_text, y_tr, y_te = load_reuters_single_label()
        describe("Reuters", y_tr, y_te)
        X_tr, X_te = vectorize(tr_text, te_text)
        new_rows, preds = run_split(X_tr, X_te, y_tr, y_te, "Reuters", 0)
        rows += new_rows
        save_predictions("Reuters", y_te, preds)
        print("Reuters done", flush=True)
    if "bydate" in args.settings:
        tr_text, te_text, y_tr, y_te = load_20ng("bydate")
        X_tr, X_te = vectorize(tr_text, te_text)
        new_rows, preds = run_split(X_tr, X_te, y_tr, y_te, "20NG bydate", 0)
        rows += new_rows
        save_predictions("20NG bydate", y_te, preds)
        print("20NG bydate done", flush=True)

    RESULTS.mkdir(exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / args.out, index=False)
    summarize(df)
    print(f"\nSaved results/{args.out}")


if __name__ == "__main__":
    main()
