"""
Null simulation: how much bias against a small class is there when the classes are identical?

Every class draws its documents from the SAME word distribution, so a perfect classifier has
nothing to go on and an unbiased one assigns test documents to classes in proportion
1 / n_classes. Any systematic preference for (or against) the small class is bias.

  vocabulary    V = 5000, true theta Zipf with exponent 1.1 (normalised)
  documents     200 tokens each, multinomial
  training      small class n_small documents, every other class 500 documents
  test          2000 documents from the same theta
  classifiers   raw counts only (no transforms):
                MNB Laplace alpha = 1 and 0.01, MNB JM lam = 0.1, 0.01, 0.001,
                CNB Laplace alpha = 1, CNB JM lam = 0.1

Two scenarios:
  two-class   the design asked for (small class vs one large class, ideal share 0.5).
              With two classes the complement of A is B, so CNB makes exactly the same
              decisions as MNB at the same smoothing; its rows are kept as a check.
  five-class  one small class and four large ones (ideal share 0.2). Here each complement
              pools three or four classes, so CNB genuinely differs from MNB.

Reading: under JM every class gets the same smoothing share, so what remains of the
deviation from the ideal share is the estimation (concavity / variance) effect of Sec. 3.1;
the additional deviation under Laplace is the effect of unequal smoothing.

Run from the repository root:  python -m src.gap_simulation [--seeds 20]
Writes results/gap/null_simulation.csv, null_simulation_runs.csv and figures/null_simulation.png.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                      # write files without opening windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse

from src.jm_smoothing import SmoothedTWCNB

OUT = Path("results") / "gap"
FIGURES = OUT / "figures"
V = 5000
ZIPF_EXPONENT = 1.1
DOC_LENGTH = 200
N_LARGE = 500
N_TEST = 2000
SMALL_SIZES = [5, 10, 20, 50, 100, 500]
SCENARIOS = {"two-class": 2, "five-class": 5}
RAW = dict(log_tf=False, idf=False, length_norm=False, weight_norm=False)

CLASSIFIERS = [   # (label, complement, smoothing, parameter)
    ("MNB a=1",      False, "laplace", 1.0),
    ("MNB a=0.01",   False, "laplace", 0.01),
    ("MNB-JM 0.1",   False, "jm",      0.1),
    ("MNB-JM 0.01",  False, "jm",      0.01),
    ("MNB-JM 0.001", False, "jm",      0.001),
    ("CNB a=1",      True,  "laplace", 1.0),
    ("CNB-JM 0.1",   True,  "jm",      0.1),
]


def zipf_theta(V=V, exponent=ZIPF_EXPONENT):
    theta = np.arange(1, V + 1, dtype=float) ** -exponent
    return theta / theta.sum()


def make_classifier(complement, smoothing, param):
    if smoothing == "jm":
        return SmoothedTWCNB(smoothing="jm", lam=param, complement=complement, **RAW)
    return SmoothedTWCNB(smoothing="laplace", alpha=param, complement=complement, **RAW)


def run(seeds):
    theta = zipf_theta()
    rows = []
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        # One test set and one pool of large-class documents per seed, shared by all settings.
        X_test = sparse.csr_matrix(rng.multinomial(DOC_LENGTH, theta, size=N_TEST))
        large = [sparse.csr_matrix(rng.multinomial(DOC_LENGTH, theta, size=N_LARGE))
                 for _ in range(max(SCENARIOS.values()) - 1)]
        small_pool = sparse.csr_matrix(rng.multinomial(DOC_LENGTH, theta, size=max(SMALL_SIZES)))
        for scenario, n_classes in SCENARIOS.items():
            for n_small in SMALL_SIZES:
                # Class 0 is the small class: the first n_small documents of the pool.
                X_train = sparse.vstack([small_pool[:n_small]] + large[:n_classes - 1]).tocsr()
                y_train = np.repeat(np.arange(n_classes), [n_small] + [N_LARGE] * (n_classes - 1))
                for label, complement, smoothing, param in CLASSIFIERS:
                    pred = make_classifier(complement, smoothing, param).fit(
                        X_train, y_train).predict(X_test)
                    rows.append({"scenario": scenario, "n_classes": n_classes, "seed": seed,
                                 "n_small": n_small, "n_large": N_LARGE, "classifier": label,
                                 "ideal_share": 1.0 / n_classes,
                                 "share_to_small": float(np.mean(pred == 0))})
        print(f"seed {seed + 1}/{seeds} done", flush=True)
    return pd.DataFrame(rows)


def summarise(runs):
    keys = ["scenario", "n_classes", "n_small", "n_large", "classifier", "ideal_share"]
    out = runs.groupby(keys, sort=False)["share_to_small"].agg(["mean", "std", "count"]).reset_index()
    return out.rename(columns={"mean": "share_mean", "std": "share_std", "count": "n_seeds"})


def null_simulation_figure(summary, path):
    """Share of test documents given to the small class against its training size."""
    scenarios = list(dict.fromkeys(summary["scenario"]))
    fig, axes = plt.subplots(1, len(scenarios), figsize=(6 * len(scenarios), 4.5), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, scenario in zip(axes, scenarios):
        sub = summary[summary["scenario"] == scenario]
        for label in dict.fromkeys(sub["classifier"]):
            curve = sub[sub["classifier"] == label]
            style = "--" if label.startswith("CNB") else "-"
            ax.errorbar(curve["n_small"], curve["share_mean"], yerr=curve["share_std"].fillna(0),
                        marker="o", capsize=3, linestyle=style, label=label)
        ax.axhline(sub["ideal_share"].iloc[0], color="black", linewidth=1, linestyle=":",
                   label="unbiased")
        ax.set_xscale("log")
        ax.set_xlabel(f"Training documents in the small class (others: {N_LARGE})")
        ax.set_title(f"Null simulation, {scenario}")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Share of test documents assigned to the small class")
    axes[-1].legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, default=20)
    args = parser.parse_args()

    runs = run(args.seeds)
    summary = summarise(runs)
    OUT.mkdir(parents=True, exist_ok=True)
    runs.to_csv(OUT / "null_simulation_runs.csv", index=False)
    summary.to_csv(OUT / "null_simulation.csv", index=False)
    null_simulation_figure(summary, FIGURES / "null_simulation.png")
    table = summary.pivot_table(index=["scenario", "n_small"], columns="classifier",
                                values="share_mean", sort=False)
    print("\nMean share of test documents assigned to the small class:")
    print(table[[c[0] for c in CLASSIFIERS]].round(4).to_string())
    print("\nSaved results/gap/null_simulation.csv and results/gap/figures/null_simulation.png")


if __name__ == "__main__":
    main()
