"""
Does unequal smoothing, rather than the concavity of log, cause the skewed-data bias?

Rennie et al. (Sec. 3.1) attribute the bias against small classes to E[log theta_hat] <
log E[theta_hat] and fix it with Complement NB. But their additive smoothing (alpha = 1 for
every class) also makes small classes' estimates mostly pseudo-counts, and the complement
removes that imbalance too. Jelinek-Mercer (JM) smoothing (src.jm_smoothing) gives every
class the same smoothing share lam, so it removes the smoothing imbalance while leaving the
concavity bias in place. Comparing MNB-JM with Complement NB separates the two causes.

Part A, imbalance  (results/gap/imbalance_jm.csv, imbalance_lambda_sweep.csv)
    The protocol of src.imbalance, re-implemented here so the original stays untouched:
    20 Newsgroups chronological split; for each seed, half the classes (rng.choice) are shrunk
    to a fraction of their training documents (each shrunk class shuffled once, so smaller
    fractions are subsets of larger ones); the vectoriser is refitted on the kept documents;
    the test set is untouched. Fixed smoothing only, no tuning. Seeds 0-2 are checked against
    the committed results/imbalance.csv for the Laplace models.
    A lambda sweep at fraction 0.1 for MNB-JM, CNB-JM, T-MNB-JM and TCNB-JM.

Part B, benchmarks  (results/gap/benchmarks_jm.csv, reuters_per_class_recall.csv)
    20 Newsgroups random (10 stratified 80/20 splits), Reuters single-label, 20 Newsgroups
    chronological. Six variants x four protocols:
      laplace_paper  alpha = 1
      laplace_tuned  alpha from ALPHA_GRID
      jm_fixed       lam = LAM_FIXED = 0.1, declared before any result was seen
      jm_tuned       lam from LAM_GRID
    Tuning uses src.experiments.pick_best (3-fold stratified cross-validation on the training
    data only) with the metric src.experiments uses for that setting (TUNE_METRIC), so that
    the Laplace rows are directly comparable with results/final_results.csv, which is checked.

Run from the repository root:
    python -m src.gap_experiment --part all
    python -m src.gap_experiment --part imbalance --seeds 2      # quick trial
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, f1_score, recall_score

from src.data import load_20ng, load_reuters_single_label
from src.experiments import ALPHAS, TUNE_METRIC, pick_best, vectorize
from src.jm_smoothing import make_smoothed_variant
from src.twcnb import VARIANTS

OUT = Path("results") / "gap"
LAM_FIXED = 0.1                       # pre-declared, not tuned
LAM_GRID = [0.5, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003]
ALPHA_GRID = list(ALPHAS)             # [1.0, 0.3, 0.1, 0.03, 0.01, 0.003]
FRACTIONS = [1.0, 0.5, 0.2, 0.1, 0.05]
SWEEP_FRACTION = 0.1
SWEEP_VARIANTS = ["MNB", "CNB", "T-MNB", "TCNB"]

IMBALANCE_MODELS = [   # (label, variant, smoothing, parameter)
    ("MNB a=1", "MNB", "laplace", 1.0), ("MNB a=0.01", "MNB", "laplace", 0.01),
    ("MNB-JM", "MNB", "jm", LAM_FIXED),
    ("CNB a=1", "CNB", "laplace", 1.0), ("CNB-JM", "CNB", "jm", LAM_FIXED),
    ("WCNB a=1", "WCNB", "laplace", 1.0), ("WCNB-JM", "WCNB", "jm", LAM_FIXED),
    ("T-MNB a=1", "T-MNB", "laplace", 1.0), ("T-MNB-JM", "T-MNB", "jm", LAM_FIXED),
    ("TCNB a=1", "TCNB", "laplace", 1.0), ("TCNB-JM", "TCNB", "jm", LAM_FIXED),
    ("TWCNB a=1", "TWCNB", "laplace", 1.0), ("TWCNB-JM", "TWCNB", "jm", LAM_FIXED),
]
# Names the committed results/imbalance.csv uses for the same Laplace models.
LEGACY_IMBALANCE_NAMES = {"MNB": "MNB a=1", "MNB a=0.01": "MNB a=0.01", "CNB": "CNB a=1",
                          "WCNB": "WCNB a=1", "TWCNB": "TWCNB a=1"}
REUTERS_RECALL_MODELS = [("MNB a=1", "MNB", "laplace", 1.0), ("CNB a=1", "CNB", "laplace", 1.0),
                         ("MNB-JM", "MNB", "jm", LAM_FIXED), ("CNB-JM", "CNB", "jm", LAM_FIXED)]


def model(variant, smoothing, param):
    """A variant with Laplace pseudo-count param, or JM weight param."""
    if smoothing == "jm":
        return make_smoothed_variant(variant, smoothing="jm", lam=param)
    return make_smoothed_variant(variant, smoothing="laplace", alpha=param)


def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


# --------------------------------------------------------------------------------------------
# Part A: imbalance
# --------------------------------------------------------------------------------------------

def shrunk_training_sets(y_train_all, seed):
    """src.imbalance's sampling: yields (fraction, keep mask, shrunk classes) for one seed."""
    classes = np.unique(y_train_all)
    rng = np.random.default_rng(seed)
    shrunk = rng.choice(classes, size=len(classes) // 2, replace=False)
    # Shuffle each shrunk class once, so smaller fractions are subsets of larger ones
    order = {c: rng.permutation(np.flatnonzero(y_train_all == c)) for c in shrunk}
    for frac in FRACTIONS:
        keep = np.ones(len(y_train_all), dtype=bool)
        for idx in order.values():
            keep[idx[max(1, int(round(frac * len(idx)))):]] = False
        yield frac, keep, shrunk


def imbalance_row(seed, frac, label, variant, smoothing, param, fitted, y_test, pred, shrunk,
                  unshrunk):
    return {"seed": seed, "fraction": frac, "model": label, "variant": variant,
            "smoothing": smoothing, "param": param,
            "accuracy": accuracy_score(y_test, pred), "macro_f1": macro_f1(y_test, pred),
            "shrunk_class_recall": recall_score(y_test, pred, labels=shrunk, average="macro",
                                                zero_division=0),
            "unshrunk_class_recall": recall_score(y_test, pred, labels=unshrunk,
                                                  average="macro", zero_division=0),
            "smoothing_share_min": float(fitted.smoothing_share_.min()),
            "smoothing_share_max": float(fitted.smoothing_share_.max())}


def run_imbalance(seeds):
    train_texts, test_texts, y_train_all, y_test = load_20ng("bydate")
    train_texts = np.array(train_texts, dtype=object)
    y_train_all, y_test = np.asarray(y_train_all), np.asarray(y_test)
    classes = np.unique(y_train_all)

    rows, sweep = [], []
    for seed in range(seeds):
        for frac, keep, shrunk in shrunk_training_sets(y_train_all, seed):
            unshrunk = np.setdiff1d(classes, shrunk)
            vectorizer = CountVectorizer()
            X_train = vectorizer.fit_transform(train_texts[keep])
            X_test = vectorizer.transform(test_texts)
            y_train = y_train_all[keep]

            for label, variant, smoothing, param in IMBALANCE_MODELS:
                fitted = model(variant, smoothing, param).fit(X_train, y_train)
                rows.append(imbalance_row(seed, frac, label, variant, smoothing, param, fitted,
                                          y_test, fitted.predict(X_test), shrunk, unshrunk))
            if frac == SWEEP_FRACTION:
                for variant in SWEEP_VARIANTS:
                    for lam in LAM_GRID:
                        fitted = model(variant, "jm", lam).fit(X_train, y_train)
                        sweep.append(imbalance_row(seed, frac, f"{variant}-JM", variant, "jm",
                                                   lam, fitted, y_test, fitted.predict(X_test),
                                                   shrunk, unshrunk))
            print(f"imbalance: seed {seed}, fraction {frac:.2f} done", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(sweep)


def check_imbalance(df, legacy_path=Path("results") / "imbalance.csv"):
    """Seeds present in the committed imbalance.csv must be reproduced by the Laplace models."""
    if not legacy_path.exists():
        print(f"(no {legacy_path} to check against)")
        return 0
    legacy = pd.read_csv(legacy_path)
    if "protocol" in legacy.columns:          # newer schema: only alpha = 1 rows are comparable
        legacy = legacy[legacy["protocol"] == "paper"]
    legacy = legacy[legacy["model"].isin(LEGACY_IMBALANCE_NAMES)].copy()
    legacy["model"] = legacy["model"].map(LEGACY_IMBALANCE_NAMES)
    cols = ["accuracy", "macro_f1", "shrunk_class_recall"]
    merged = legacy.merge(df, on=["seed", "fraction", "model"], suffixes=("_ref", ""))
    bad = 0
    for c in cols:
        off = merged[(merged[c] - merged[c + "_ref"]).abs() > 5e-5]
        bad += len(off)
        for _, r in off.iterrows():
            print(f"  MISMATCH imbalance seed {r.seed} frac {r.fraction} {r.model} {c}: "
                  f"{r[c]:.4f} vs committed {r[c + '_ref']:.4f}")
    print(f"imbalance check: {len(merged)} committed rows compared, {bad} mismatching values")
    return bad


# --------------------------------------------------------------------------------------------
# Part B: benchmarks
# --------------------------------------------------------------------------------------------

def timed_fit_predict(estimator, X_tr, y_tr, X_te):
    start = time.perf_counter()
    estimator.fit(X_tr, y_tr)
    fit_time = time.perf_counter() - start
    return estimator, estimator.predict(X_te), fit_time


def run_benchmark_split(X_tr, X_te, y_tr, y_te, setting, split):
    metric = TUNE_METRIC[setting]
    rows = []
    for variant in VARIANTS:
        alpha = pick_best(lambda v: model(variant, "laplace", v), ALPHA_GRID, X_tr, y_tr, split,
                          metric)
        lam = pick_best(lambda v: model(variant, "jm", v), LAM_GRID, X_tr, y_tr, split, metric)
        for protocol, smoothing, param in [("laplace_paper", "laplace", 1.0),
                                           ("laplace_tuned", "laplace", alpha),
                                           ("jm_fixed", "jm", LAM_FIXED),
                                           ("jm_tuned", "jm", lam)]:
            _, pred, fit_time = timed_fit_predict(model(variant, smoothing, param), X_tr, y_tr, X_te)
            rows.append({"setting": setting, "split": split, "variant": variant,
                         "protocol": protocol, "smoothing": smoothing, "param": param,
                         "accuracy": accuracy_score(y_te, pred), "macro_f1": macro_f1(y_te, pred),
                         "fit_time_s": fit_time})
    return rows


def reuters_per_class_recall(X_tr, X_te, y_tr, y_te):
    classes, train_docs = np.unique(y_tr, return_counts=True)
    table = pd.DataFrame({"class": classes, "train_docs": train_docs})
    for label, variant, smoothing, param in REUTERS_RECALL_MODELS:
        pred = model(variant, smoothing, param).fit(X_tr, y_tr).predict(X_te)
        table[label] = recall_score(y_te, pred, labels=classes, average=None, zero_division=0)
    table = table.sort_values("train_docs").reset_index(drop=True)
    # As in the notebook: small half = classes with at most the median training count.
    table["half"] = np.where(table["train_docs"] <= table["train_docs"].median(), "small", "large")
    return table


def run_benchmarks(n_splits):
    rows, recall_table = [], None
    for split in range(n_splits):
        tr, te, y_tr, y_te = load_20ng("random", seed=split)
        X_tr, X_te = vectorize(tr, te)
        start = time.perf_counter()
        rows += run_benchmark_split(X_tr, X_te, y_tr, y_te, "20NG random", split)
        print(f"benchmarks: 20NG random split {split + 1}/{n_splits} done "
              f"in {time.perf_counter() - start:.0f}s", flush=True)

    tr, te, y_tr, y_te = load_reuters_single_label()
    X_tr, X_te = vectorize(tr, te)
    rows += run_benchmark_split(X_tr, X_te, y_tr, y_te, "Reuters", 0)
    recall_table = reuters_per_class_recall(X_tr, X_te, y_tr, y_te)
    print("benchmarks: Reuters done", flush=True)

    tr, te, y_tr, y_te = load_20ng("bydate")
    X_tr, X_te = vectorize(tr, te)
    rows += run_benchmark_split(X_tr, X_te, y_tr, y_te, "20NG bydate", 0)
    print("benchmarks: 20NG bydate done", flush=True)
    return pd.DataFrame(rows), recall_table


def check_benchmarks(df, reference=Path("results") / "final_results.csv"):
    """laplace_paper / laplace_tuned rows against a final_results.csv (to 4 decimals)."""
    ref = pd.read_csv(reference)
    ref = ref[ref["model"] != "LinearSVC"].rename(columns={"model": "variant"})
    ref["protocol"] = "laplace_" + ref["protocol"]
    keys = ["setting", "split", "variant", "protocol"]
    merged = ref.merge(df, on=keys, suffixes=("_ref", ""))
    bad = 0
    for _, r in merged.iterrows():
        problems = [c for c in ["accuracy", "macro_f1"] if abs(r[c] - r[c + "_ref"]) > 5e-5]
        if abs(r["param"] - r["param_ref"]) > 1e-12:
            problems.append("param")
        if problems:
            bad += 1
            print(f"  MISMATCH {r.setting} split {r.split} {r.variant} {r.protocol}: "
                  f"param {r.param:g} vs {r.param_ref:g}, acc {r.accuracy:.4f} vs "
                  f"{r.accuracy_ref:.4f}, macro-F1 {r.macro_f1:.4f} vs {r.macro_f1_ref:.4f}")
    by_protocol = merged.groupby("protocol").size().to_dict()
    print(f"benchmark check against {reference}: {len(merged)} rows compared "
          f"({by_protocol}), {bad} mismatching rows")
    return bad


# --------------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--part", choices=["imbalance", "benchmarks", "all"], default="all")
    parser.add_argument("--seeds", type=int, default=10, help="imbalance seeds (0 .. seeds-1)")
    parser.add_argument("--n-splits", type=int, default=10, help="random 20 Newsgroups splits")
    parser.add_argument("--reference", default=str(Path("results") / "final_results.csv"),
                        help="final_results.csv to check the Laplace benchmark rows against")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    if args.part in ("imbalance", "all"):
        df, sweep = run_imbalance(args.seeds)
        df.to_csv(OUT / "imbalance_jm.csv", index=False)
        sweep.to_csv(OUT / "imbalance_lambda_sweep.csv", index=False)
        check_imbalance(df)
        for metric in ["shrunk_class_recall", "macro_f1"]:
            table = df.pivot_table(index="fraction", columns="model", values=metric, sort=False)
            print(f"\nmean {metric} by fraction kept:")
            print(table.sort_index(ascending=False)[[m[0] for m in IMBALANCE_MODELS]]
                  .round(4).T.to_string())
        print("\nSaved results/gap/imbalance_jm.csv and imbalance_lambda_sweep.csv")

    if args.part in ("benchmarks", "all"):
        df, recall_table = run_benchmarks(args.n_splits)
        df.to_csv(OUT / "benchmarks_jm.csv", index=False)
        recall_table.to_csv(OUT / "reuters_per_class_recall.csv", index=False)
        check_benchmarks(df, Path(args.reference))
        metric_of = {"20NG random": "accuracy", "Reuters": "macro_f1", "20NG bydate": "accuracy"}
        for setting, metric in metric_of.items():
            table = df[df["setting"] == setting].pivot_table(
                index="variant", columns="protocol", values=metric, sort=False)
            print(f"\n{setting}, mean {metric}:")
            print(table.reindex(list(VARIANTS)).round(4).to_string())
        print("\nReuters mean recall by half (split at median training count):")
        print(recall_table.groupby("half")[[m[0] for m in REUTERS_RECALL_MODELS]].mean()
              .round(3).to_string())
        print("\nSaved results/gap/benchmarks_jm.csv and reuters_per_class_recall.csv")


if __name__ == "__main__":
    main()
