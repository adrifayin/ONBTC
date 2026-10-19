"""
Paired significance tests across the 10 random 20 Newsgroups splits.

Random 80/20 splits share most of their training data, so the ordinary paired
t-test is over-confident. We also report the corrected resampled t-test of
Nadeau & Bengio (2003), which multiplies the variance by (1/k + n_test/n_train)
instead of 1/k.
Run from the repository root (needs results/final_results.csv):
    python -m src.significance
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RESULTS = Path("results")
TEST_TRAIN_RATIO = 0.2 / 0.8

COMPARISONS = [   # (protocol, model A, model B, what the comparison tests)
    ("paper", "CNB",       "MNB",   "complement, alpha = 1"),
    ("tuned", "CNB",       "MNB",   "complement, tuned alpha"),
    ("paper", "WCNB",      "CNB",   "weight normalisation, raw counts"),
    ("paper", "TWCNB",     "TCNB",  "weight normalisation, transformed"),
    ("tuned", "T-MNB",     "MNB",   "transforms, tuned alpha"),
    ("tuned", "TCNB",      "T-MNB", "complement on top of transforms, tuned"),
    ("paper", "LinearSVC", "TWCNB", "SVM vs the paper's model"),
    ("tuned", "LinearSVC", "T-MNB", "SVM vs the best tuned Naive Bayes"),
]


def corrected_ttest(diff):
    k = len(diff)
    t = diff.mean() / np.sqrt((1 / k + TEST_TRAIN_RATIO) * diff.var(ddof=1))
    return 2 * stats.t.sf(abs(t), df=k - 1)


def main():
    df = pd.read_csv(RESULTS / "final_results.csv")
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
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "significance.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()