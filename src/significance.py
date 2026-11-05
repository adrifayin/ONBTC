"""
Significance tests and confidence intervals.

1. 20 Newsgroups, 10 random splits: paired t-tests across splits. Random 80/20 splits
   share most of their training data, so the ordinary paired t-test is over-confident;
   we also report the corrected resampled t-test of Nadeau & Bengio (2003), which
   multiplies the variance by (1/k + n_test/n_train) instead of 1/k.
2. Reuters and 20 Newsgroups chronological (one fixed test set each): paired bootstrap
   over test documents (Koehn 2004; Berg-Kirkpatrick et al. 2012). Each resample draws
   test documents with replacement and recomputes both models' scores on the same draw.
   Also writes a 95% percentile interval for every model's score.

Run from the repository root (needs results/final_results.csv and results/predictions/):
    python -m src.significance
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.experiments import metric_score

RESULTS = Path("results")
PREDICTIONS = RESULTS / "predictions"
TEST_TRAIN_RATIO = 0.2 / 0.8
N_BOOT = 2000
SINGLE_SPLIT = ["Reuters", "20NG bydate"]

COMPARISONS = [   # (protocol, model A, model B, what the comparison tests)
    ("paper", "CNB",       "MNB",   "complement, alpha = 1"),
    ("tuned", "CNB",       "MNB",   "complement, tuned alpha"),
    ("paper", "WCNB",      "CNB",   "weight normalisation, raw counts"),
    ("paper", "TWCNB",     "TCNB",  "weight normalisation, transformed"),
    ("tuned", "TWCNB",     "TCNB",  "weight normalisation, transformed, tuned"),
    ("tuned", "T-MNB",     "MNB",   "transforms, tuned alpha"),
    ("tuned", "TCNB",      "T-MNB", "complement on top of transforms, tuned"),
    ("paper", "LinearSVC", "TWCNB", "SVM vs the paper's model"),
    ("tuned", "LinearSVC", "T-MNB", "SVM vs tuned T-MNB"),
    ("tuned", "LinearSVC", "TCNB",  "SVM vs tuned TCNB"),
]


def corrected_ttest(diff):
    k = len(diff)
    t = diff.mean() / np.sqrt((1 / k + TEST_TRAIN_RATIO) * diff.var(ddof=1))
    return 2 * stats.t.sf(abs(t), df=k - 1)


def random_split_tests(df):
    df = df[df["setting"] == "20NG random"]
    rows = []
    for metric in ["accuracy", "macro_f1"]:
        for protocol, a, b, label in COMPARISONS:
            table = df[df["protocol"] == protocol].pivot(index="split", columns="model", values=metric)
            diff = (table[a] - table[b]).to_numpy()
            rows.append({
                "metric": metric,
                "comparison": f"{a} vs {b} ({protocol})",
                "tests": label,
                "mean_diff_points": round(100 * diff.mean(), 2),
                "A_better_in": f"{int((diff > 0).sum())}/{len(diff)} splits",
                "p_paired": round(stats.ttest_rel(table[a], table[b]).pvalue, 4),
                "p_corrected": round(corrected_ttest(diff), 4),
            })
    return pd.DataFrame(rows)


def bootstrap_indices(n, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(N_BOOT, n))


def single_split_tests():
    """Paired bootstrap for each comparison, and a 95% interval for every model."""
    tests, intervals = [], []
    for setting in SINGLE_SPLIT:
        path = PREDICTIONS / f"{setting.replace(' ', '_').lower()}.npz"
        if not path.exists():
            print(f"skipping {setting}: {path} not found (run python -m src.experiments)")
            continue
        saved = np.load(path, allow_pickle=True)
        y = saved["y_true"]
        preds = {k: saved[k] for k in saved.files if k != "y_true"}
        idx = bootstrap_indices(len(y))
        for metric in ["accuracy", "macro_f1"]:
            boot = {k: np.array([metric_score(metric, y[i], p[i]) for i in idx])
                    for k, p in preds.items()}
            for key, p in preds.items():
                protocol, model = key.split("|")
                lo, hi = np.percentile(boot[key], [2.5, 97.5])
                intervals.append({"setting": setting, "metric": metric, "protocol": protocol,
                                  "model": model, "score": round(metric_score(metric, y, p), 4),
                                  "ci_low": round(lo, 4), "ci_high": round(hi, 4)})
            for protocol, a, b, label in COMPARISONS:
                ka, kb = f"{protocol}|{a}", f"{protocol}|{b}"
                observed = metric_score(metric, y, preds[ka]) - metric_score(metric, y, preds[kb])
                diff = boot[ka] - boot[kb]
                lo, hi = np.percentile(diff, [2.5, 97.5])
                # two-sided: how often the resampled difference falls on the other side of zero
                p = min(1.0, 2 * min((diff <= 0).mean(), (diff >= 0).mean()))
                tests.append({"setting": setting, "metric": metric,
                              "comparison": f"{a} vs {b} ({protocol})", "tests": label,
                              "diff_points": round(100 * observed, 2),
                              "ci95_points": f"[{100 * lo:.2f}, {100 * hi:.2f}]",
                              "p_bootstrap": round(p, 4)})
    return pd.DataFrame(tests), pd.DataFrame(intervals)


def main():
    df = pd.read_csv(RESULTS / "final_results.csv")
    out = random_split_tests(df)
    out.to_csv(RESULTS / "significance.csv", index=False)
    print("20 Newsgroups, 10 random splits:")
    print(out.to_string(index=False))

    tests, intervals = single_split_tests()
    if len(tests):
        tests.to_csv(RESULTS / "significance_bootstrap.csv", index=False)
        intervals.to_csv(RESULTS / "bootstrap_ci.csv", index=False)
        print(f"\nSingle-split settings, paired bootstrap ({N_BOOT} resamples of the test set):")
        print(tests.to_string(index=False))
        print("\nSaved results/significance_bootstrap.csv and results/bootstrap_ci.csv")


if __name__ == "__main__":
    main()
