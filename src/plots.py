"""
Figures for the report, blog and slides. Run after the experiments:  python -m src.plots
Reads results/final_results.csv, and results/imbalance.csv if it exists.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                      # write files without opening windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS = Path("results")
FIGURES = RESULTS / "figures"
ORDER = ["MNB", "CNB", "WCNB", "T-MNB", "TCNB", "TWCNB", "LinearSVC"]
SETTINGS = {"20NG random": "accuracy", "Reuters": "macro_f1", "20NG bydate": "accuracy"}


def results_figure(df, setting, metric):
    stats = df[df["setting"] == setting].groupby(["protocol", "model"])[metric].agg(["mean", "std"])
    x = np.arange(len(ORDER))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, protocol in enumerate(["paper", "tuned"]):
        s = stats.loc[protocol].reindex(ORDER)
        ax.bar(x + (i - 0.5) * width, s["mean"], width, yerr=s["std"].fillna(0),
               capsize=3, label=f"{protocol} protocol")
    ax.set_ylim(max(0.0, stats["mean"].min() - 0.05), min(1.0, stats["mean"].max() + 0.03))
    ax.set_xticks(x)
    ax.set_xticklabels(ORDER, rotation=20)
    ax.set_ylabel("Accuracy" if metric == "accuracy" else "Macro-F1")
    ax.set_title(setting)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / f"results_{setting.replace(' ', '_').lower()}.png", dpi=200)
    plt.close(fig)


def timing_figure(df):
    sub = df[(df["setting"] == "20NG random") & (df["protocol"] == "paper")]
    times = sub.groupby("model")["fit_time_s"].mean().reindex(ORDER)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(ORDER, times)
    ax.set_yscale("log")
    for i, t in enumerate(times):
        ax.text(i, t, f"{t:.2f}s", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Training time (s, log scale)")
    ax.set_title("Training time on 20 Newsgroups (mean of 10 splits)")
    plt.setp(ax.get_xticklabels(), rotation=20)
    fig.tight_layout()
    fig.savefig(FIGURES / "training_time.png", dpi=200)
    plt.close(fig)


def imbalance_figure(df):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    panels = [(axes[0], "macro_f1", "Macro-F1 (all classes)"),
              (axes[1], "shrunk_class_recall", "Mean recall on shrunk classes")]
    fractions = sorted(df["fraction"].unique(), reverse=True)
    for ax, metric, title in panels:
        for model in df["model"].unique():
            curve = df[df["model"] == model].groupby("fraction")[metric].agg(["mean", "std"])
            ax.errorbar(curve.index, curve["mean"], yerr=curve["std"].fillna(0),
                        marker="o", capsize=3, label=model)
        ax.set_xscale("log")
        ax.minorticks_off()
        ax.set_xticks(fractions)
        ax.set_xticklabels([f"{int(f * 100)}%" for f in fractions])
        ax.invert_xaxis()                  # more imbalance further to the right
        ax.set_xlabel("Training documents kept for shrunk classes")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Score")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "imbalance.png", dpi=200)
    plt.close(fig)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RESULTS / "final_results.csv")
    for setting, metric in SETTINGS.items():
        if setting in set(df["setting"]):
            results_figure(df, setting, metric)
    timing_figure(df)
    if (RESULTS / "imbalance.csv").exists():
        imbalance_figure(pd.read_csv(RESULTS / "imbalance.csv"))
    print("Saved figures to results/figures/")


if __name__ == "__main__":
    main()