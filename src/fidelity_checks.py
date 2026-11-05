"""
How literally can Table 4 of Rennie et al. (2003) be followed? Three checks on the
random 20 Newsgroups splits (the paper's protocol):

  step6 : Table 4, step 6 as printed divides by the plain sum, not by the sum of
          absolute values (Eq. 8). The weights flip sign and arg min picks the worst class.
  step7 : Table 4, step 7 says the test document is the raw count vector. We compare
          that with applying the training transforms (log TF, IDF) to test documents.
          Length-normalising a test document only rescales its scores, so it never
          changes the arg min; the difference comes from log TF and IDF.
  svm   : the paper used SvmFu (standard hinge loss); our baseline uses scikit-learn's
          default squared hinge. We train both at the paper's C = 10.

Run from the repository root:  python -m src.fidelity_checks
"""
import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

from src.data import load_20ng
from src.experiments import vectorize
from src.twcnb import RennieTransformer, make_variant

RESULTS = Path("results")


def literal_step6_predict(model, X):
    """Predict with Table 4, step 6 exactly as printed: w / sum(w), no absolute value."""
    w = model.w_ / model.w_.sum(axis=1, keepdims=True)
    T = model.transformer_.transform(X)
    return model.classes_[np.asarray(T @ w.T).argmin(axis=1)]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-splits", type=int, default=10)
    parser.add_argument("--svm-splits", type=int, default=3,
                        help="splits for the hinge-loss SVM (about 2 minutes each)")
    args = parser.parse_args()

    rows = []
    for split in range(args.n_splits):
        tr_text, te_text, y_tr, y_te = load_20ng("random", seed=split)
        X_tr, X_te = vectorize(tr_text, te_text)

        def add(check, variant, setting, alpha, pred, fit_time=np.nan):
            rows.append({"split": split, "check": check, "model": variant, "setting": setting,
                         "alpha": alpha, "accuracy": accuracy_score(y_te, pred),
                         "fit_time_s": fit_time})

        for variant in ["WCNB", "TWCNB"]:
            normed = make_variant(variant).fit(X_tr, y_tr)
            add("step6", variant, "Eq. 8 (|.|)", 1.0, normed.predict(X_te))
            plain = make_variant(variant, weight_norm=False).fit(X_tr, y_tr)
            add("step6", variant, "Table 4 as printed", 1.0, literal_step6_predict(plain, X_te))

        for variant in ["T-MNB", "TCNB", "TWCNB"]:
            for alpha in [1.0, 0.1]:
                for transform_test, label in [(True, "transformed test docs"),
                                              (False, "raw-count test docs (as printed)")]:
                    model = make_variant(variant, alpha=alpha,
                                         transform_test=transform_test).fit(X_tr, y_tr)
                    add("step7", variant, label, alpha, model.predict(X_te))

        if split < args.svm_splits:
            for loss in ["squared_hinge", "hinge"]:
                svm = make_pipeline(RennieTransformer(),
                                    LinearSVC(C=10.0, loss=loss, dual=True, max_iter=20000))
                start = time.perf_counter()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    svm.fit(X_tr, y_tr)
                add("svm", "LinearSVC", loss, np.nan, svm.predict(X_te),
                    time.perf_counter() - start)
        print(f"split {split + 1}/{args.n_splits} done", flush=True)

    RESULTS.mkdir(exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "fidelity_checks.csv", index=False)
    summary = (df.groupby(["check", "model", "alpha", "setting"], dropna=False)
                 [["accuracy", "fit_time_s"]].agg(["mean", "std", "count"]).round(4))
    print(summary.to_string())


if __name__ == "__main__":
    main()
