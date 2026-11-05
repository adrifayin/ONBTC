"""
Statistics, figures and the written summary for the JM-smoothing experiment.

Reads results/gap/*.csv (from src.gap_experiment and src.gap_simulation) and, read-only,
results/final_results.csv. Writes, all inside results/gap/:

  significance_jm.csv       paired tests (20NG random splits; imbalance seeds at 10 %)
  recovery_ratio.csv        R = (MNB-JM - MNB a=1) / (CNB a=1 - MNB a=1) with seed-bootstrap CIs
  effective_smoothing.csv   pseudo-count ratio r = alpha * V / N_c behind every tuned alpha
  figures/*.png             imbalance_jm, lambda_sweep, null_simulation, benchmarks_jm,
                            smoothing_share
  SUMMARY.md                tables, decision-rule outcome, limitations

Run from the repository root (after the experiments):  python -m src.gap_analysis
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                      # write files without opening windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.data import load_20ng, load_reuters_single_label
from src.experiments import vectorize
from src.gap_experiment import LAM_FIXED, LAM_GRID, SWEEP_FRACTION, model
from src.gap_simulation import null_simulation_figure
from src.significance import corrected_ttest
from src.twcnb import VARIANTS

RESULTS = Path("results")
OUT = RESULTS / "gap"
FIGURES = OUT / "figures"
N_BOOT = 2000
R_FRACTIONS = [0.5, 0.2, 0.1, 0.05]
PRIMARY_FRACTION = 0.1
METRIC_OF = {"20NG random": "accuracy", "Reuters": "macro_f1", "20NG bydate": "accuracy"}
PROTOCOLS = ["laplace_paper", "laplace_tuned", "jm_fixed", "jm_tuned"]

RANDOM_COMPARISONS = [   # (A variant, A protocol, B variant, B protocol, question)
    ("MNB", "jm_fixed", "CNB", "laplace_paper", "can equal smoothing replace the complement?"),
    ("CNB", "jm_fixed", "MNB", "jm_fixed", "complement effect at equal smoothing"),
    ("CNB", "laplace_paper", "MNB", "laplace_paper", "the paper's comparison (alpha = 1)"),
    ("T-MNB", "jm_fixed", "T-MNB", "laplace_tuned", "does untuned lam = 0.1 match tuned alpha?"),
    ("TWCNB", "jm_fixed", "TWCNB", "laplace_paper", "JM vs alpha = 1 for the paper's model"),
    ("TCNB", "jm_fixed", "T-MNB", "jm_fixed", "complement on top of transforms, equal smoothing"),
]
IMBALANCE_COMPARISONS = [
    ("MNB-JM", "CNB a=1", "can equal smoothing replace the complement?"),
    ("CNB-JM", "MNB-JM", "complement effect at equal smoothing"),
]


def label_of(variant, protocol):
    return f"{variant} {protocol}"


# --------------------------------------------------------------------------------------------
# 1. Significance
# --------------------------------------------------------------------------------------------

def significance(bench, imb):
    rows = []
    rand = bench[bench["setting"] == "20NG random"]
    for metric in ["accuracy", "macro_f1"]:
        table = rand.pivot_table(index="split", columns=["variant", "protocol"], values=metric)
        for va, pa, vb, pb, question in RANDOM_COMPARISONS:
            a, b = table[(va, pa)].to_numpy(), table[(vb, pb)].to_numpy()
            diff = a - b
            rows.append({"family": "20NG random (10 splits)", "metric": metric,
                         "comparison": f"{label_of(va, pa)} vs {label_of(vb, pb)}",
                         "question": question, "mean_diff_points": 100 * diff.mean(),
                         "A_better_in": f"{int((diff > 0).sum())}/{len(diff)}",
                         "p_paired_t": stats.ttest_rel(a, b).pvalue,
                         "p_corrected_t": corrected_ttest(diff), "p_wilcoxon": np.nan})
    sub = imb[imb["fraction"] == PRIMARY_FRACTION]
    table = sub.pivot_table(index="seed", columns="model", values="shrunk_class_recall")
    for a_name, b_name, question in IMBALANCE_COMPARISONS:
        a, b = table[a_name].to_numpy(), table[b_name].to_numpy()
        diff = a - b
        rows.append({"family": f"imbalance, fraction {PRIMARY_FRACTION} ({len(diff)} seeds)",
                     "metric": "shrunk_class_recall", "comparison": f"{a_name} vs {b_name}",
                     "question": question, "mean_diff_points": 100 * diff.mean(),
                     "A_better_in": f"{int((diff > 0).sum())}/{len(diff)}",
                     "p_paired_t": stats.ttest_rel(a, b).pvalue, "p_corrected_t": np.nan,
                     "p_wilcoxon": stats.wilcoxon(a, b).pvalue})
    out = pd.DataFrame(rows)
    # Bonferroni over every comparison in the file, applied to its primary p-value: the
    # corrected resampled t-test for the overlapping random splits, the paired t-test for
    # the independent imbalance seeds (Wilcoxon adjusted alongside as a check).
    m = len(out)
    primary = out["p_corrected_t"].fillna(out["p_paired_t"])
    out["p_primary"] = primary
    out["p_bonferroni"] = np.minimum(1.0, primary * m)
    out["p_wilcoxon_bonferroni"] = np.minimum(1.0, out["p_wilcoxon"] * m)
    out["n_tests"] = m
    return out


# --------------------------------------------------------------------------------------------
# 2. Recovery ratio
# --------------------------------------------------------------------------------------------

def recovery(table):
    """R from a (seed x model) table of shrunk-class recall."""
    mnb, jm, cnb = (table[c].mean() for c in ["MNB a=1", "MNB-JM", "CNB a=1"])
    return (jm - mnb) / (cnb - mnb)


def decision(R):
    if R >= 0.8:
        return "R >= 0.8: smoothing imbalance explains most of the skewed-data bias"
    if R >= 0.3:
        return "0.3 <= R < 0.8: both causes matter"
    return "R < 0.3: the paper's concavity explanation dominates"


def recovery_ratios(imb):
    rng = np.random.default_rng(0)
    rows = []
    for frac in R_FRACTIONS:
        table = imb[imb["fraction"] == frac].pivot_table(index="seed", columns="model",
                                                         values="shrunk_class_recall")
        R = recovery(table)
        seeds = table.index.to_numpy()
        boot = np.array([recovery(table.loc[rng.choice(seeds, size=len(seeds))])
                         for _ in range(N_BOOT)])
        lo, hi = np.nanpercentile(boot, [2.5, 97.5])
        rows.append({"fraction": frac, "MNB a=1": table["MNB a=1"].mean(),
                     "MNB-JM": table["MNB-JM"].mean(), "CNB a=1": table["CNB a=1"].mean(),
                     "R": R, "ci_low": lo, "ci_high": hi, "n_seeds": len(seeds),
                     "case": decision(R)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------
# 3. Effective smoothing
# --------------------------------------------------------------------------------------------

def setting_data(setting, split):
    if setting == "20NG random":
        tr, te, y_tr, _ = load_20ng("random", seed=split)
    elif setting == "Reuters":
        tr, te, y_tr, _ = load_reuters_single_label()
    else:
        tr, te, y_tr, _ = load_20ng("bydate")
    return vectorize(tr, te)[0], np.asarray(y_tr)


def effective_smoothing(final, bench):
    """r_c = alpha * V / N_c for every tuned Laplace pick, from both sources of picks."""
    picks = [final[(final["protocol"] == "tuned") & (final["model"] != "LinearSVC")]
             .rename(columns={"model": "variant"}).assign(source="final_results.csv")]
    if bench is not None:
        picks.append(bench[bench["protocol"] == "laplace_tuned"].assign(source="benchmarks_jm.csv"))
    picks = pd.concat(picks)[["source", "setting", "split", "variant", "param"]]
    rows = []
    for (setting, split), group in picks.groupby(["setting", "split"], sort=False):
        X, y = setting_data(setting, split)
        V = X.shape[1]
        for _, p in group.iterrows():
            fitted = model(p["variant"], "laplace", p["param"]).fit(X, y)
            mass = fitted.class_mass_
            r = p["param"] * V / mass
            rows.append({"source": p["source"], "setting": setting, "split": split,
                         "variant": p["variant"], "alpha": p["param"], "V": V,
                         "mean_class_mass": mass.mean(), "alpha_V": p["param"] * V,
                         "r_median": np.median(r), "r_min": r.min(), "r_max": r.max()})
        print(f"effective smoothing: {setting} split {split} done", flush=True)
    return pd.DataFrame(rows)


def agreement(eff):
    """Spread across the six variants of log10 alpha vs log10 median r, per split."""
    rows = []
    for (source, setting), g in eff.groupby(["source", "setting"], sort=False):
        per_split = g.groupby("split").agg(
            sd_log_alpha=("alpha", lambda s: np.log10(s).std(ddof=0)),
            sd_log_r=("r_median", lambda s: np.log10(s).std(ddof=0)))
        rows.append({"source": source, "setting": setting,
                     "sd_log10_alpha": per_split["sd_log_alpha"].mean(),
                     "sd_log10_r": per_split["sd_log_r"].mean(), "n_splits": len(per_split)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------
# 4. Figures
# --------------------------------------------------------------------------------------------

COLOURS = {"MNB": "C0", "CNB": "C1", "WCNB": "C4", "T-MNB": "C2", "TCNB": "C3", "TWCNB": "C5"}


def imbalance_figure(imb):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fractions = sorted(imb["fraction"].unique(), reverse=True)
    for ax, (metric, title) in zip(axes, [("shrunk_class_recall", "Mean recall on shrunk classes"),
                                          ("macro_f1", "Macro-F1 (all classes)")]):
        for variant in ["MNB", "CNB", "TWCNB"]:
            for name, style in [(f"{variant} a=1", "-"), (f"{variant}-JM", "--")]:
                curve = imb[imb["model"] == name].groupby("fraction")[metric].agg(["mean", "std"])
                ax.errorbar(curve.index, curve["mean"], yerr=curve["std"].fillna(0), marker="o",
                            capsize=3, linestyle=style, color=COLOURS[variant],
                            label=name.replace("-JM", f"-JM (lam={LAM_FIXED:g})"))
        ax.set_xscale("log")
        ax.minorticks_off()
        ax.set_xticks(fractions)
        ax.set_xticklabels([f"{int(f * 100)}%" for f in fractions])
        ax.invert_xaxis()                  # more imbalance further to the right
        ax.set_xlabel("Training documents kept for shrunk classes")
        ax.set_title(f"{title}: Laplace (solid) vs JM (dashed)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Score (mean ± std over seeds)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "imbalance_jm.png", dpi=200)
    plt.close(fig)


def lambda_sweep_figure(sweep, imb):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ref = imb[imb["fraction"] == SWEEP_FRACTION]
    for ax, (metric, title) in zip(axes, [("shrunk_class_recall", "Mean recall on shrunk classes"),
                                          ("macro_f1", "Macro-F1 (all classes)")]):
        for variant in dict.fromkeys(sweep["variant"]):
            curve = sweep[sweep["variant"] == variant].groupby("param")[metric].agg(["mean", "std"])
            ax.errorbar(curve.index, curve["mean"], yerr=curve["std"].fillna(0), marker="o",
                        capsize=3, color=COLOURS[variant], label=f"{variant}-JM")
        for name, style in [("CNB a=1", ":"), ("MNB a=1", "-.")]:
            ax.axhline(ref[ref["model"] == name][metric].mean(), color="grey", linestyle=style,
                       label=name)
        ax.axvline(LAM_FIXED, color="black", linewidth=0.8, alpha=0.5)
        ax.set_xscale("log")
        ax.set_xlabel("JM weight lam (vertical line: pre-declared 0.1)")
        ax.set_title(f"{title}, {int(SWEEP_FRACTION * 100)}% of shrunk-class documents kept")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Score (mean ± std over seeds)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "lambda_sweep.png", dpi=200)
    plt.close(fig)


def benchmarks_figure(bench):
    settings = list(METRIC_OF)
    fig, axes = plt.subplots(1, len(settings), figsize=(17, 4.5))
    x = np.arange(len(VARIANTS))
    width = 0.2
    for ax, setting in zip(axes, settings):
        metric = METRIC_OF[setting]
        st = bench[bench["setting"] == setting].groupby(["protocol", "variant"])[metric].agg(
            ["mean", "std"])
        for i, protocol in enumerate(PROTOCOLS):
            s = st.loc[protocol].reindex(list(VARIANTS))
            ax.bar(x + (i - 1.5) * width, s["mean"], width, yerr=s["std"].fillna(0), capsize=2,
                   label=protocol)
        ax.set_ylim(max(0.0, st["mean"].min() - 0.05), min(1.0, st["mean"].max() + 0.03))
        ax.set_xticks(x)
        ax.set_xticklabels(list(VARIANTS), rotation=20)
        ax.set_ylabel("Accuracy" if metric == "accuracy" else "Macro-F1")
        ax.set_title(setting)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "benchmarks_jm.png", dpi=200)
    plt.close(fig)


def smoothing_share_figure():
    """Per-class pseudo-count share on Reuters: Laplace alpha = 1 (raw and transformed) vs JM."""
    X, y = setting_data("Reuters", 0)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for variant in ["MNB", "CNB", "T-MNB"]:
        fitted = model(variant, "laplace", 1.0).fit(X, y)
        ax.scatter(fitted.class_count_, fitted.smoothing_share_, s=18, color=COLOURS[variant],
                   label=f"{variant}, Laplace alpha = 1")
    ax.axhline(LAM_FIXED, color="black", linestyle="--", label=f"any variant, JM lam = {LAM_FIXED:g}")
    ax.set_xscale("log")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Training documents in the class (Reuters)")
    ax.set_ylabel("Share of the estimate from smoothing")
    ax.set_title("How much of each class's estimate is pseudo-counts")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "smoothing_share.png", dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------------------------
# 5. SUMMARY.md
# --------------------------------------------------------------------------------------------

def md_table(df, floatfmt="{:.4f}"):
    def cell(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else floatfmt.format(v)
        return str(v)
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def mean_std_table(df, index, columns, value, index_order=None, column_order=None):
    g = df.groupby([index, columns])[value]
    mean, std = g.mean().unstack(), g.std().unstack()
    if index_order is not None:
        mean, std = mean.reindex(index_order), std.reindex(index_order)
    if column_order is not None:
        mean, std = mean[column_order], std[column_order]
    out = mean.copy().astype(object)
    for c in mean.columns:
        out[c] = [f"{m:.4f}" if np.isnan(s) else f"{m:.4f} ± {s:.4f}"
                  for m, s in zip(mean[c], std[c])]
    return out.reset_index()


def pfmt(p):
    return "" if np.isnan(p) else (f"{p:.2e}" if p < 1e-4 else f"{p:.4f}")


def write_summary(imb, sweep, bench, recall, sig, rr, eff, agree, null):
    L = []
    add = L.append
    primary = rr[rr["fraction"] == PRIMARY_FRACTION].iloc[0]
    n_seeds = imb["seed"].nunique()

    add("# Is skewed-data bias a smoothing artefact? Jelinek-Mercer vs Complement NB\n")
    add("## The gap\n")
    add("Rennie et al. (2003, Sec. 3.1) attribute the bias of multinomial Naive Bayes against "
        "classes with few training documents to the concavity of log "
        "(E[log θ̂] < log E[θ̂]) and correct it with Complement NB. The same paper smooths every "
        "class additively with one α = 1, which adds α·V pseudo-counts to every class whatever "
        "its size: a small class's estimate is then mostly pseudo-counts, and the Rennie "
        "transforms (which shrink class totals by length-normalising documents) make this worse. "
        "Complement NB removes this smoothing imbalance as well, because every complement has "
        "roughly the same mass, so the paper's experiments cannot tell the two causes apart. "
        "Here additive smoothing is replaced by Jelinek-Mercer (JM) interpolation toward the "
        "uniform distribution, θ_ci = (1 − λ)·N_ci/N_c + λ/V, which gives every class the same "
        "smoothing share λ and is invariant to rescaling the counts, but leaves the concavity "
        "effect in place. If MNB with JM recovers what Complement NB gains, the gain was mostly "
        "a smoothing effect.\n")
    add("Related work: Frank & Bouckaert (PKDD 2006) proposed a per-class prior adjustment for "
        "multinomial NB on unbalanced data, and Liu & Martin (MLDM 2011) studied smoothing under "
        "class imbalance. The JM fix is therefore not claimed as new; the contribution is the "
        "cause-separating comparison against Complement NB, with and without Rennie's "
        "transforms.\n")

    add("## Pre-declared choices\n")
    add(f"- Fixed JM weight **λ = {LAM_FIXED:g}**, chosen before any result was seen and never "
        "tuned on test data. The tuned protocol picks λ from "
        f"{LAM_GRID} and α from the existing grid by 3-fold cross-validation on the training "
        "data (src.experiments.pick_best, accuracy for 20 Newsgroups, macro-F1 for Reuters, "
        "as in src.experiments).")
    add("- Decision rule, on mean shrunk-class recall at 10 % of shrunk-class documents kept: "
        "R = (MNB-JM − MNB α=1) / (CNB α=1 − MNB α=1). R ≥ 0.8: smoothing imbalance explains "
        "most of the skewed-data bias; 0.3 ≤ R < 0.8: both causes matter; R < 0.3: the paper's "
        "concavity explanation dominates.\n")

    add("## Headline: recovery ratio\n")
    add(f"At {int(PRIMARY_FRACTION * 100)} %: **R = {primary['R']:.3f}** (95 % seed-bootstrap "
        f"CI [{primary['ci_low']:.3f}, {primary['ci_high']:.3f}], {int(primary['n_seeds'])} "
        f"seeds) → **{primary['case']}**.\n")
    show = rr.copy()
    show["95% CI"] = [f"[{lo:.3f}, {hi:.3f}]" for lo, hi in zip(show["ci_low"], show["ci_high"])]
    add(md_table(show[["fraction", "MNB a=1", "MNB-JM", "CNB a=1", "R", "95% CI", "case"]]))
    add("\n(MNB a=1, MNB-JM and CNB a=1 columns: mean shrunk-class recall.)\n")

    add(f"## Imbalance stress test (20 Newsgroups chronological, {n_seeds} seeds)\n")
    order = list(dict.fromkeys(imb["model"]))
    fr = sorted(imb["fraction"].unique(), reverse=True)
    for metric, title in [("shrunk_class_recall", "Mean recall on the shrunk classes"),
                          ("unshrunk_class_recall", "Mean recall on the untouched classes"),
                          ("macro_f1", "Macro-F1")]:
        add(f"### {title} (mean ± std over seeds), by fraction of shrunk-class documents kept\n")
        t = mean_std_table(imb, "model", "fraction", metric, index_order=order, column_order=fr)
        t.columns = ["model"] + [f"{int(f * 100)}%" for f in fr]
        add(md_table(t) + "\n")
    add(f"### λ sweep at {int(SWEEP_FRACTION * 100)} % (shrunk-class recall, mean ± std)\n")
    t = mean_std_table(sweep, "model", "param", "shrunk_class_recall")
    t = t[["model"] + sorted([c for c in t.columns if c != "model"], reverse=True)]
    t.columns = ["model"] + [f"λ={c:g}" for c in t.columns[1:]]
    add(md_table(t) + "\n")
    ref = imb[imb["fraction"] == SWEEP_FRACTION]
    add(f"For reference at {int(SWEEP_FRACTION * 100)} %: CNB a=1 "
        f"{ref[ref['model'] == 'CNB a=1']['shrunk_class_recall'].mean():.4f}, MNB a=0.01 "
        f"{ref[ref['model'] == 'MNB a=0.01']['shrunk_class_recall'].mean():.4f}.\n")

    add("## Benchmarks (mean ± std; 20NG random over 10 splits, others single split)\n")
    for setting, metric in METRIC_OF.items():
        sub = bench[bench["setting"] == setting]
        add(f"### {setting}: {metric}\n")
        t = mean_std_table(sub, "variant", "protocol", metric, index_order=list(VARIANTS),
                           column_order=PROTOCOLS)
        add(md_table(t) + "\n")
        params = sub.pivot_table(index="variant", columns="protocol", values="param",
                                 aggfunc=lambda s: ", ".join(f"{v:g}" for v in sorted(set(s))))
        params = params.reindex(list(VARIANTS))[["laplace_tuned", "jm_tuned"]].reset_index()
        add("Tuned values chosen: " + "; ".join(
            f"{r.variant}: α {r.laplace_tuned}, λ {r.jm_tuned}" for r in params.itertuples()) + "\n")

    add("### Reuters: mean recall by half of the classes (split at the median training count)\n")
    halves = recall.groupby("half")[["MNB a=1", "CNB a=1", "MNB-JM", "CNB-JM"]].mean()
    counts = recall.groupby("half").size()
    halves.insert(0, "classes", counts)
    add(md_table(halves.reindex(["small", "large"]).reset_index(), "{:.3f}") + "\n")
    add("(Notebook values for comparison: MNB 0.004 / 0.440, CNB 0.320 / 0.654.)\n")

    add("## Significance\n")
    add(f"Bonferroni over all {int(sig['n_tests'].iloc[0])} comparisons below, applied to the "
        "primary p-value (Nadeau–Bengio corrected t-test for the overlapping random splits; "
        "paired t-test for the independent imbalance seeds).\n")
    s = sig.copy()
    for c in ["p_paired_t", "p_corrected_t", "p_wilcoxon", "p_bonferroni"]:
        s[c] = s[c].map(pfmt)
    s["mean_diff_points"] = s["mean_diff_points"].map(lambda v: f"{v:+.2f}")
    add(md_table(s[["family", "metric", "comparison", "question", "mean_diff_points",
                    "A_better_in", "p_paired_t", "p_corrected_t", "p_wilcoxon",
                    "p_bonferroni"]]) + "\n")

    add("## Effective smoothing behind the tuned α\n")
    add("r = α·V / N_c, with N_c the class mass the estimate is built from (own class, or "
        "complement, after the variant's transforms); median over classes, averaged over "
        "splits. If smoothing strength is really about the pseudo-count ratio, the tuned "
        "values should agree across variants better in r than in α.\n")
    e = (eff.groupby(["source", "setting", "variant"], sort=False)
            .agg(alpha=("alpha", lambda s: ", ".join(f"{v:g}" for v in sorted(set(s)))),
                 mean_class_mass=("mean_class_mass", "mean"), alpha_V=("alpha_V", "mean"),
                 r_median=("r_median", "mean"), r_min=("r_min", "min"), r_max=("r_max", "max"))
            .reset_index())
    add(md_table(e, "{:.4g}") + "\n")
    add("Spread across the six variants (std of log10, mean over splits; smaller = more "
        "agreement):\n")
    add(md_table(agree) + "\n")

    add("## Null simulation (identical classes, raw counts)\n")
    add("Share of test documents assigned to the small class (mean ± std over seeds); the "
        "unbiased value is 1/(number of classes). In the two-class scenario the complement of "
        "one class is the other class, so CNB makes exactly the same decisions as MNB at the "
        "same smoothing; the five-class scenario (one small, four large classes) is added so "
        "that CNB differs.\n")
    nt = null.copy()
    nt["cell"] = [f"{m:.4f} ± {s:.4f}" for m, s in zip(nt["share_mean"], nt["share_std"])]
    for scenario in dict.fromkeys(nt["scenario"]):
        sub = nt[nt["scenario"] == scenario]
        t = sub.pivot_table(index="n_small", columns="classifier", values="cell", aggfunc="first",
                            sort=False).reset_index()
        add(f"### {scenario} (unbiased share {sub['ideal_share'].iloc[0]:.2f})\n")
        add(md_table(t) + "\n")

    add("## Limitations\n")
    add("- Reuters and the chronological 20 Newsgroups split are single fixed splits; their "
        "rows carry no between-split variance, and per-split differences there are not tested.")
    add(f"- λ = {LAM_FIXED:g} was fixed a priori; the λ sweep shows how much depends on it.")
    add("- JM interpolates toward the uniform distribution, not toward a corpus background "
        "model; a background model is the more common choice in language-model retrieval and "
        "could behave differently.")
    add("- The imbalance stress test shrinks half the classes of one dataset (20 Newsgroups); "
        "the recovery ratio is defined on that design only.")
    add("- The 20NG random splits overlap; the corrected t-test accounts for this only "
        "approximately.")
    add("- The committed results/final_results.csv was tuned with an earlier protocol for some "
        "tuned rows; this experiment re-tunes with the current src.experiments.pick_best.\n")

    add("## What this means for the paper's Sec. 3.1 claim\n")
    add(interpretation(rr, sig, null))
    (OUT / "SUMMARY.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def interpretation(rr, sig, null):
    """Statements that follow mechanically from the numbers; nothing beyond them."""
    p = rr[rr["fraction"] == PRIMARY_FRACTION].iloc[0]
    imb_sig = sig[sig["family"].str.startswith("imbalance")].set_index("comparison")
    jm_vs_cnb = imb_sig.loc["MNB-JM vs CNB a=1"]
    cnb_vs_jm = imb_sig.loc["CNB-JM vs MNB-JM"]

    def verdict(row):
        if row["p_bonferroni"] < 0.05:
            return ("significantly higher" if row["mean_diff_points"] > 0
                    else "significantly lower")
        return "not significantly different (after Bonferroni)"

    parts = [
        f"On the stress test at {int(PRIMARY_FRACTION * 100)} %, replacing α = 1 by JM "
        f"(λ = {LAM_FIXED:g}) in plain MNB recovers R = {p['R']:.2f} of the gain that Complement "
        f"NB brings (CI [{p['ci_low']:.2f}, {p['ci_high']:.2f}]); by the pre-declared rule: "
        f"{p['case']}.",
        f"MNB-JM's shrunk-class recall is {verdict(jm_vs_cnb)} than CNB α = 1's "
        f"({jm_vs_cnb['mean_diff_points']:+.2f} points), and adding the complement on top of "
        f"equal smoothing (CNB-JM vs MNB-JM) is {verdict(cnb_vs_jm)} "
        f"({cnb_vs_jm['mean_diff_points']:+.2f} points).",
    ]
    rand = sig[sig["family"].str.startswith("20NG")]
    rand = rand[rand["metric"] == "accuracy"].set_index("comparison")
    cnb_mnb_jm = rand.loc["CNB jm_fixed vs MNB jm_fixed"]
    paper = rand.loc["CNB laplace_paper vs MNB laplace_paper"]
    parts.append(
        f"On the 20 Newsgroups random splits the complement is worth "
        f"{paper['mean_diff_points']:+.2f} accuracy points at α = 1 but "
        f"{cnb_mnb_jm['mean_diff_points']:+.2f} points when both models use JM "
        f"({verdict(cnb_mnb_jm)}).")
    two = null[(null["scenario"] == "two-class") & (null["classifier"].str.startswith("MNB-JM"))]
    smallest = two[two["n_small"] == two["n_small"].min()]
    parts.append(
        f"The null simulation shows that the bias against a small class does not need unequal "
        f"smoothing: with identical classes and JM, a class with {int(smallest['n_small'].iloc[0])} "
        f"training documents receives at most {smallest['share_mean'].max():.3f} of the test "
        f"documents (unbiased: 0.5). In that setting the estimation effect of Sec. 3.1 alone is "
        f"enough to produce strong bias, so the two causes are not mutually exclusive.")
    parts.append(
        "These results speak to one dataset design and one fixed λ; they do not show that the "
        "concavity argument is wrong, only how much of Complement NB's measured benefit on "
        "these benchmarks can also be obtained by equalising the smoothing.")
    return " ".join(parts) + "\n"


# --------------------------------------------------------------------------------------------

def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    imb = pd.read_csv(OUT / "imbalance_jm.csv")
    sweep = pd.read_csv(OUT / "imbalance_lambda_sweep.csv")
    bench = pd.read_csv(OUT / "benchmarks_jm.csv")
    recall = pd.read_csv(OUT / "reuters_per_class_recall.csv")
    null = pd.read_csv(OUT / "null_simulation.csv")
    final = pd.read_csv(RESULTS / "final_results.csv")

    sig = significance(bench, imb)
    sig.to_csv(OUT / "significance_jm.csv", index=False)
    rr = recovery_ratios(imb)
    rr.to_csv(OUT / "recovery_ratio.csv", index=False)
    eff = effective_smoothing(final, bench)
    eff.to_csv(OUT / "effective_smoothing.csv", index=False)
    agree = agreement(eff)

    imbalance_figure(imb)
    lambda_sweep_figure(sweep, imb)
    null_simulation_figure(null, FIGURES / "null_simulation.png")
    benchmarks_figure(bench)
    smoothing_share_figure()
    write_summary(imb, sweep, bench, recall, sig, rr, eff, agree, null)

    print(rr[["fraction", "R", "ci_low", "ci_high", "case"]].round(3).to_string(index=False))
    print(sig[["family", "metric", "comparison", "mean_diff_points", "A_better_in",
               "p_primary", "p_bonferroni"]].round(4).to_string(index=False))
    print(agree.round(3).to_string(index=False))
    print("\nSaved results/gap/SUMMARY.md, significance_jm.csv, recovery_ratio.csv, "
          "effective_smoothing.csv and results/gap/figures/")


if __name__ == "__main__":
    main()
