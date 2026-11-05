# Is skewed-data bias a smoothing artefact? Jelinek-Mercer vs Complement NB

## The gap

Rennie et al. (2003, Sec. 3.1) attribute the bias of multinomial Naive Bayes against classes with few training documents to the concavity of log (E[log θ̂] < log E[θ̂]) and correct it with Complement NB. The same paper smooths every class additively with one α = 1, which adds α·V pseudo-counts to every class whatever its size: a small class's estimate is then mostly pseudo-counts, and the Rennie transforms (which shrink class totals by length-normalising documents) make this worse. Complement NB removes this smoothing imbalance as well, because every complement has roughly the same mass, so the paper's experiments cannot tell the two causes apart. Here additive smoothing is replaced by Jelinek-Mercer (JM) interpolation toward the uniform distribution, θ_ci = (1 − λ)·N_ci/N_c + λ/V, which gives every class the same smoothing share λ and is invariant to rescaling the counts, but leaves the concavity effect in place. If MNB with JM recovers what Complement NB gains, the gain was mostly a smoothing effect.

Related work: Frank & Bouckaert (PKDD 2006) proposed a per-class prior adjustment for multinomial NB on unbalanced data, and Liu & Martin (MLDM 2011) studied smoothing under class imbalance. The JM fix is therefore not claimed as new; the contribution is the cause-separating comparison against Complement NB, with and without Rennie's transforms.

## Pre-declared choices

- Fixed JM weight **λ = 0.1**, chosen before any result was seen and never tuned on test data. The tuned protocol picks λ from [0.5, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003] and α from the existing grid by 3-fold cross-validation on the training data (src.experiments.pick_best, accuracy for 20 Newsgroups, macro-F1 for Reuters, as in src.experiments).
- Decision rule, on mean shrunk-class recall at 10 % of shrunk-class documents kept: R = (MNB-JM − MNB α=1) / (CNB α=1 − MNB α=1). R ≥ 0.8: smoothing imbalance explains most of the skewed-data bias; 0.3 ≤ R < 0.8: both causes matter; R < 0.3: the paper's concavity explanation dominates.

## Headline: recovery ratio

At 10 %: **R = 1.215** (95 % seed-bootstrap CI [1.029, 1.479], 10 seeds) → **R >= 0.8: smoothing imbalance explains most of the skewed-data bias**.

| fraction | MNB a=1 | MNB-JM | CNB a=1 | R | 95% CI | case |
|---|---|---|---|---|---|---|
| 0.5000 | 0.1892 | 0.5799 | 0.5358 | 1.1274 | [1.052, 1.208] | R >= 0.8: smoothing imbalance explains most of the skewed-data bias |
| 0.2000 | 0.0271 | 0.3792 | 0.3066 | 1.2597 | [1.113, 1.446] | R >= 0.8: smoothing imbalance explains most of the skewed-data bias |
| 0.1000 | 0.0073 | 0.1995 | 0.1655 | 1.2155 | [1.029, 1.479] | R >= 0.8: smoothing imbalance explains most of the skewed-data bias |
| 0.0500 | 0.0027 | 0.0776 | 0.0833 | 0.9296 | [0.782, 1.154] | R >= 0.8: smoothing imbalance explains most of the skewed-data bias |

(MNB a=1, MNB-JM and CNB a=1 columns: mean shrunk-class recall.)

## Imbalance stress test (20 Newsgroups chronological, 10 seeds)

### Mean recall on the shrunk classes (mean ± std over seeds), by fraction of shrunk-class documents kept

| model | 100% | 50% | 20% | 10% | 5% |
|---|---|---|---|---|---|
| MNB a=1 | 0.5327 ± 0.0602 | 0.1892 ± 0.0435 | 0.0271 ± 0.0124 | 0.0073 ± 0.0059 | 0.0027 ± 0.0023 |
| MNB a=0.01 | 0.6501 ± 0.0460 | 0.5812 ± 0.0497 | 0.4087 ± 0.0535 | 0.2594 ± 0.0419 | 0.1322 ± 0.0289 |
| MNB-JM | 0.6508 ± 0.0478 | 0.5799 ± 0.0501 | 0.3792 ± 0.0522 | 0.1995 ± 0.0396 | 0.0776 ± 0.0243 |
| CNB a=1 | 0.6680 ± 0.0583 | 0.5358 ± 0.0668 | 0.3066 ± 0.0779 | 0.1655 ± 0.0610 | 0.0833 ± 0.0383 |
| CNB-JM | 0.6655 ± 0.0581 | 0.5315 ± 0.0681 | 0.2965 ± 0.0802 | 0.1539 ± 0.0612 | 0.0746 ± 0.0365 |
| WCNB a=1 | 0.6533 ± 0.0641 | 0.4893 ± 0.0631 | 0.2478 ± 0.0639 | 0.1189 ± 0.0408 | 0.0547 ± 0.0240 |
| WCNB-JM | 0.6632 ± 0.0609 | 0.5259 ± 0.0642 | 0.2946 ± 0.0724 | 0.1536 ± 0.0566 | 0.0762 ± 0.0347 |
| T-MNB a=1 | 0.6269 ± 0.0442 | 0.1375 ± 0.0284 | 0.0088 ± 0.0036 | 0.0027 ± 0.0020 | 0.0016 ± 0.0017 |
| T-MNB-JM | 0.6827 ± 0.0338 | 0.5911 ± 0.0384 | 0.3782 ± 0.0463 | 0.1995 ± 0.0371 | 0.0853 ± 0.0254 |
| TCNB a=1 | 0.7034 ± 0.0399 | 0.5211 ± 0.0520 | 0.2093 ± 0.0579 | 0.0784 ± 0.0331 | 0.0262 ± 0.0160 |
| TCNB-JM | 0.6955 ± 0.0385 | 0.5865 ± 0.0484 | 0.3903 ± 0.0653 | 0.2472 ± 0.0647 | 0.1428 ± 0.0626 |
| TWCNB a=1 | 0.6990 ± 0.0413 | 0.4821 ± 0.0558 | 0.1611 ± 0.0504 | 0.0510 ± 0.0226 | 0.0148 ± 0.0087 |
| TWCNB-JM | 0.6936 ± 0.0408 | 0.5967 ± 0.0471 | 0.4201 ± 0.0610 | 0.2818 ± 0.0638 | 0.1747 ± 0.0640 |

### Mean recall on the untouched classes (mean ± std over seeds), by fraction of shrunk-class documents kept

| model | 100% | 50% | 20% | 10% | 5% |
|---|---|---|---|---|---|
| MNB a=1 | 0.5244 ± 0.0602 | 0.6130 ± 0.0578 | 0.6294 ± 0.0607 | 0.6340 ± 0.0609 | 0.6371 ± 0.0613 |
| MNB a=0.01 | 0.6216 ± 0.0460 | 0.6502 ± 0.0458 | 0.6755 ± 0.0469 | 0.6865 ± 0.0471 | 0.6917 ± 0.0482 |
| MNB-JM | 0.6205 ± 0.0478 | 0.6501 ± 0.0465 | 0.6775 ± 0.0465 | 0.6870 ± 0.0474 | 0.6924 ± 0.0486 |
| CNB a=1 | 0.6387 ± 0.0583 | 0.6859 ± 0.0564 | 0.6998 ± 0.0568 | 0.7027 ± 0.0568 | 0.7027 ± 0.0578 |
| CNB-JM | 0.6361 ± 0.0581 | 0.6836 ± 0.0561 | 0.6976 ± 0.0564 | 0.7004 ± 0.0573 | 0.7015 ± 0.0574 |
| WCNB a=1 | 0.6256 ± 0.0641 | 0.6710 ± 0.0610 | 0.6781 ± 0.0637 | 0.6792 ± 0.0642 | 0.6791 ± 0.0648 |
| WCNB-JM | 0.6337 ± 0.0609 | 0.6781 ± 0.0583 | 0.6896 ± 0.0596 | 0.6921 ± 0.0604 | 0.6921 ± 0.0616 |
| T-MNB a=1 | 0.6260 ± 0.0442 | 0.6852 ± 0.0432 | 0.6865 ± 0.0433 | 0.6863 ± 0.0432 | 0.6864 ± 0.0434 |
| T-MNB-JM | 0.6681 ± 0.0338 | 0.7002 ± 0.0339 | 0.7252 ± 0.0350 | 0.7342 ± 0.0370 | 0.7384 ± 0.0371 |
| TCNB a=1 | 0.6925 ± 0.0399 | 0.7379 ± 0.0379 | 0.7485 ± 0.0382 | 0.7478 ± 0.0391 | 0.7475 ± 0.0392 |
| TCNB-JM | 0.6825 ± 0.0385 | 0.7213 ± 0.0380 | 0.7429 ± 0.0377 | 0.7493 ± 0.0372 | 0.7512 ± 0.0378 |
| TWCNB a=1 | 0.6876 ± 0.0413 | 0.7341 ± 0.0388 | 0.7416 ± 0.0394 | 0.7409 ± 0.0407 | 0.7402 ± 0.0403 |
| TWCNB-JM | 0.6778 ± 0.0408 | 0.7123 ± 0.0400 | 0.7352 ± 0.0406 | 0.7414 ± 0.0409 | 0.7442 ± 0.0406 |

### Macro-F1 (mean ± std over seeds), by fraction of shrunk-class documents kept

| model | 100% | 50% | 20% | 10% | 5% |
|---|---|---|---|---|---|
| MNB a=1 | 0.5184 ± 0.0000 | 0.3644 ± 0.0288 | 0.2438 ± 0.0335 | 0.2258 ± 0.0335 | 0.2217 ± 0.0318 |
| MNB a=0.01 | 0.6251 ± 0.0000 | 0.6049 ± 0.0043 | 0.5250 ± 0.0131 | 0.4395 ± 0.0259 | 0.3499 ± 0.0299 |
| MNB-JM | 0.6252 ± 0.0000 | 0.6048 ± 0.0051 | 0.5099 ± 0.0149 | 0.3998 ± 0.0298 | 0.3058 ± 0.0333 |
| CNB a=1 | 0.6374 ± 0.0000 | 0.5962 ± 0.0131 | 0.4685 ± 0.0362 | 0.3746 ± 0.0409 | 0.3115 ± 0.0344 |
| CNB-JM | 0.6347 ± 0.0000 | 0.5922 ± 0.0139 | 0.4605 ± 0.0367 | 0.3643 ± 0.0410 | 0.3032 ± 0.0332 |
| WCNB a=1 | 0.6230 ± 0.0000 | 0.5696 ± 0.0155 | 0.4295 ± 0.0387 | 0.3388 ± 0.0419 | 0.2862 ± 0.0376 |
| WCNB-JM | 0.6315 ± 0.0000 | 0.5884 ± 0.0124 | 0.4589 ± 0.0351 | 0.3636 ± 0.0437 | 0.3032 ± 0.0377 |
| T-MNB a=1 | 0.6298 ± 0.0000 | 0.3681 ± 0.0233 | 0.2554 ± 0.0219 | 0.2486 ± 0.0222 | 0.2473 ± 0.0227 |
| T-MNB-JM | 0.6769 ± 0.0000 | 0.6459 ± 0.0051 | 0.5387 ± 0.0156 | 0.4243 ± 0.0223 | 0.3349 ± 0.0235 |
| TCNB a=1 | 0.6938 ± 0.0000 | 0.6234 ± 0.0140 | 0.4315 ± 0.0309 | 0.3287 ± 0.0241 | 0.2822 ± 0.0193 |
| TCNB-JM | 0.6861 ± 0.0000 | 0.6497 ± 0.0087 | 0.5481 ± 0.0243 | 0.4573 ± 0.0306 | 0.3810 ± 0.0343 |
| TWCNB a=1 | 0.6890 ± 0.0000 | 0.6009 ± 0.0179 | 0.3941 ± 0.0299 | 0.3035 ± 0.0218 | 0.2695 ± 0.0202 |
| TWCNB-JM | 0.6830 ± 0.0000 | 0.6508 ± 0.0065 | 0.5634 ± 0.0199 | 0.4783 ± 0.0270 | 0.4035 ± 0.0317 |

### λ sweep at 10 % (shrunk-class recall, mean ± std)

| model | λ=0.5 | λ=0.3 | λ=0.1 | λ=0.03 | λ=0.01 | λ=0.003 | λ=0.001 | λ=0.0003 |
|---|---|---|---|---|---|---|---|---|
| CNB-JM | 0.0692 ± 0.0340 | 0.1024 ± 0.0495 | 0.1539 ± 0.0612 | 0.1888 ± 0.0683 | 0.2068 ± 0.0680 | 0.2175 ± 0.0654 | 0.2230 ± 0.0635 | 0.2265 ± 0.0621 |
| MNB-JM | 0.3107 ± 0.0477 | 0.2649 ± 0.0467 | 0.1995 ± 0.0396 | 0.1587 ± 0.0349 | 0.1356 ± 0.0311 | 0.1174 ± 0.0283 | 0.1054 ± 0.0258 | 0.0962 ± 0.0243 |
| T-MNB-JM | 0.3147 ± 0.0395 | 0.2605 ± 0.0400 | 0.1995 ± 0.0371 | 0.1624 ± 0.0348 | 0.1409 ± 0.0322 | 0.1242 ± 0.0291 | 0.1136 ± 0.0271 | 0.1044 ± 0.0257 |
| TCNB-JM | 0.1941 ± 0.0723 | 0.2212 ± 0.0709 | 0.2472 ± 0.0647 | 0.2566 ± 0.0630 | 0.2569 ± 0.0613 | 0.2553 ± 0.0591 | 0.2537 ± 0.0582 | 0.2524 ± 0.0578 |

For reference at 10 %: CNB a=1 0.1655, MNB a=0.01 0.2594.

## Benchmarks (mean ± std; 20NG random over 10 splits, others single split)

### 20NG random: accuracy

| variant | laplace_paper | laplace_tuned | jm_fixed | jm_tuned |
|---|---|---|---|---|
| MNB | 0.8537 ± 0.0076 | 0.8918 ± 0.0072 | 0.8810 ± 0.0065 | 0.8925 ± 0.0064 |
| CNB | 0.9009 ± 0.0065 | 0.9043 ± 0.0067 | 0.8940 ± 0.0057 | 0.9025 ± 0.0072 |
| WCNB | 0.8785 ± 0.0055 | 0.8783 ± 0.0058 | 0.8803 ± 0.0054 | 0.8813 ± 0.0056 |
| T-MNB | 0.8834 ± 0.0064 | 0.9210 ± 0.0075 | 0.9184 ± 0.0072 | 0.9232 ± 0.0076 |
| TCNB | 0.9126 ± 0.0065 | 0.9145 ± 0.0058 | 0.9136 ± 0.0065 | 0.9153 ± 0.0063 |
| TWCNB | 0.9101 ± 0.0067 | 0.9127 ± 0.0068 | 0.9127 ± 0.0069 | 0.9154 ± 0.0064 |

Tuned values chosen: MNB: α 0.003, λ 0.0003, 0.001, 0.003, 0.01; CNB: α 0.3, λ 0.03; WCNB: α 0.3, 1, λ 0.03, 0.1; T-MNB: α 0.01, 0.03, λ 0.5; TCNB: α 0.1, 0.3, λ 0.3, 0.5; TWCNB: α 0.3, λ 0.5

### Reuters: macro_f1

| variant | laplace_paper | laplace_tuned | jm_fixed | jm_tuned |
|---|---|---|---|---|
| MNB | 0.2334 | 0.5625 | 0.5639 | 0.6468 |
| CNB | 0.5285 | 0.5242 | 0.5222 | 0.5237 |
| WCNB | 0.5172 | 0.6036 | 0.5520 | 0.6048 |
| T-MNB | 0.1494 | 0.6094 | 0.5577 | 0.6392 |
| TCNB | 0.4277 | 0.5378 | 0.5152 | 0.5152 |
| TWCNB | 0.3255 | 0.5861 | 0.5785 | 0.5785 |

Tuned values chosen: MNB: α 0.003, λ 0.5; CNB: α 0.3, λ 0.01; WCNB: α 0.3, λ 0.01; T-MNB: α 0.003, λ 0.5; TCNB: α 0.1, λ 0.1; TWCNB: α 0.03, λ 0.1

### 20NG bydate: accuracy

| variant | laplace_paper | laplace_tuned | jm_fixed | jm_tuned |
|---|---|---|---|---|
| MNB | 0.5429 | 0.6491 | 0.6458 | 0.6455 |
| CNB | 0.6693 | 0.6718 | 0.6665 | 0.6730 |
| WCNB | 0.6549 | 0.6617 | 0.6638 | 0.6683 |
| T-MNB | 0.6471 | 0.6990 | 0.6880 | 0.7011 |
| TCNB | 0.7153 | 0.7181 | 0.7042 | 0.7185 |
| TWCNB | 0.7114 | 0.7163 | 0.7002 | 0.7195 |

Tuned values chosen: MNB: α 0.03, λ 0.03; CNB: α 0.3, λ 0.03; WCNB: α 0.3, λ 0.03; T-MNB: α 0.03, λ 0.5; TCNB: α 0.3, λ 0.5; TWCNB: α 0.3, λ 0.5

### Reuters: mean recall by half of the classes (split at the median training count)

| half | classes | MNB a=1 | CNB a=1 | MNB-JM | CNB-JM |
|---|---|---|---|---|---|
| small | 26 | 0.004 | 0.320 | 0.339 | 0.310 |
| large | 26 | 0.440 | 0.654 | 0.770 | 0.651 |

(Notebook values for comparison: MNB 0.004 / 0.440, CNB 0.320 / 0.654.)

## Significance

Bonferroni over all 14 comparisons below, applied to the primary p-value (Nadeau–Bengio corrected t-test for the overlapping random splits; paired t-test for the independent imbalance seeds).

| family | metric | comparison | question | mean_diff_points | A_better_in | p_paired_t | p_corrected_t | p_wilcoxon | p_bonferroni |
|---|---|---|---|---|---|---|---|---|---|
| 20NG random (10 splits) | accuracy | MNB jm_fixed vs CNB laplace_paper | can equal smoothing replace the complement? | -1.99 | 0/10 | 5.40e-08 | 1.10e-05 |  | 0.0002 |
| 20NG random (10 splits) | accuracy | CNB jm_fixed vs MNB jm_fixed | complement effect at equal smoothing | +1.30 | 10/10 | 2.61e-06 | 0.0004 |  | 0.0050 |
| 20NG random (10 splits) | accuracy | CNB laplace_paper vs MNB laplace_paper | the paper's comparison (alpha = 1) | +4.72 | 10/10 | 1.85e-11 | 4.92e-09 |  | 6.89e-08 |
| 20NG random (10 splits) | accuracy | T-MNB jm_fixed vs T-MNB laplace_tuned | does untuned lam = 0.1 match tuned alpha? | -0.25 | 0/10 | 0.0002 | 0.0091 |  | 0.1273 |
| 20NG random (10 splits) | accuracy | TWCNB jm_fixed vs TWCNB laplace_paper | JM vs alpha = 1 for the paper's model | +0.26 | 9/10 | 0.0011 | 0.0331 |  | 0.4636 |
| 20NG random (10 splits) | accuracy | TCNB jm_fixed vs T-MNB jm_fixed | complement on top of transforms, equal smoothing | -0.48 | 1/10 | 0.0030 | 0.0599 |  | 0.8393 |
| 20NG random (10 splits) | macro_f1 | MNB jm_fixed vs CNB laplace_paper | can equal smoothing replace the complement? | -2.55 | 0/10 | 5.76e-08 | 1.17e-05 |  | 0.0002 |
| 20NG random (10 splits) | macro_f1 | CNB jm_fixed vs MNB jm_fixed | complement effect at equal smoothing | +1.75 | 10/10 | 4.62e-07 | 7.82e-05 |  | 0.0011 |
| 20NG random (10 splits) | macro_f1 | CNB laplace_paper vs MNB laplace_paper | the paper's comparison (alpha = 1) | +5.81 | 10/10 | 4.78e-11 | 1.25e-08 |  | 1.75e-07 |
| 20NG random (10 splits) | macro_f1 | T-MNB jm_fixed vs T-MNB laplace_tuned | does untuned lam = 0.1 match tuned alpha? | -0.17 | 0/10 | 0.0008 | 0.0258 |  | 0.3617 |
| 20NG random (10 splits) | macro_f1 | TWCNB jm_fixed vs TWCNB laplace_paper | JM vs alpha = 1 for the paper's model | +0.59 | 10/10 | 7.20e-06 | 0.0008 |  | 0.0117 |
| 20NG random (10 splits) | macro_f1 | TCNB jm_fixed vs T-MNB jm_fixed | complement on top of transforms, equal smoothing | -0.70 | 0/10 | 0.0003 | 0.0135 |  | 0.1891 |
| imbalance, fraction 0.1 (10 seeds) | shrunk_class_recall | MNB-JM vs CNB a=1 | can equal smoothing replace the complement? | +3.41 | 7/10 | 0.0537 |  | 0.0645 | 0.7522 |
| imbalance, fraction 0.1 (10 seeds) | shrunk_class_recall | CNB-JM vs MNB-JM | complement effect at equal smoothing | -4.56 | 2/10 | 0.0141 |  | 0.0371 | 0.1971 |

## Effective smoothing behind the tuned α

r = α·V / N_c, with N_c the class mass the estimate is built from (own class, or complement, after the variant's transforms); median over classes, averaged over splits. If smoothing strength is really about the pseudo-count ratio, the tuned values should agree across variants better in r than in α.

| source | setting | variant | alpha | mean_class_mass | alpha_V | r_median | r_min | r_max |
|---|---|---|---|---|---|---|---|---|
| final_results.csv | 20NG random | MNB | 0.003 | 2.195e+05 | 466.2 | 0.002148 | 0.001142 | 0.004324 |
| final_results.csv | 20NG random | CNB | 0.1, 0.3, 1 | 4.171e+06 | 6.463e+04 | 0.01548 | 0.003704 | 0.03903 |
| final_results.csv | 20NG random | WCNB | 1 | 4.171e+06 | 1.554e+05 | 0.03724 | 0.03409 | 0.03989 |
| final_results.csv | 20NG random | T-MNB | 0.003, 0.01, 0.03 | 6991 | 2695 | 0.3887 | 0.05142 | 0.9436 |
| final_results.csv | 20NG random | TCNB | 0.1, 0.3 | 1.328e+05 | 3.113e+04 | 0.2342 | 0.1125 | 0.3685 |
| final_results.csv | 20NG random | TWCNB | 0.1, 0.3 | 1.328e+05 | 3.729e+04 | 0.2806 | 0.1125 | 0.3685 |
| benchmarks_jm.csv | 20NG random | MNB | 0.003 | 2.195e+05 | 466.2 | 0.002148 | 0.001142 | 0.004324 |
| benchmarks_jm.csv | 20NG random | CNB | 0.3 | 4.171e+06 | 4.662e+04 | 0.01117 | 0.01023 | 0.01197 |
| benchmarks_jm.csv | 20NG random | WCNB | 0.3, 1 | 4.171e+06 | 1.234e+05 | 0.02954 | 0.01023 | 0.03989 |
| benchmarks_jm.csv | 20NG random | T-MNB | 0.01, 0.03 | 6991 | 3428 | 0.4944 | 0.1661 | 0.921 |
| benchmarks_jm.csv | 20NG random | TCNB | 0.1, 0.3 | 1.328e+05 | 4.34e+04 | 0.3266 | 0.1194 | 0.3633 |
| benchmarks_jm.csv | 20NG random | TWCNB | 0.3 | 1.328e+05 | 4.662e+04 | 0.3508 | 0.3246 | 0.3685 |
| final_results.csv | Reuters | MNB | 0.03 | 1.485e+04 | 713.2 | 0.2547 | 0.003128 | 8.593 |
| final_results.csv | Reuters | CNB | 0.1 | 7.573e+05 | 2377 | 0.00309 | 0.003079 | 0.004369 |
| final_results.csv | Reuters | WCNB | 0.3 | 7.573e+05 | 7132 | 0.009271 | 0.009238 | 0.01311 |
| final_results.csv | Reuters | T-MNB | 0.003 | 812.2 | 71.32 | 0.5341 | 0.004517 | 10.05 |
| final_results.csv | Reuters | TCNB | 0.1 | 4.142e+04 | 2377 | 0.05647 | 0.0563 | 0.0899 |
| final_results.csv | Reuters | TWCNB | 0.03 | 4.142e+04 | 713.2 | 0.01694 | 0.01689 | 0.02697 |
| benchmarks_jm.csv | Reuters | MNB | 0.003 | 1.485e+04 | 71.32 | 0.02547 | 0.0003128 | 0.8593 |
| benchmarks_jm.csv | Reuters | CNB | 0.3 | 7.573e+05 | 7132 | 0.009271 | 0.009238 | 0.01311 |
| benchmarks_jm.csv | Reuters | WCNB | 0.3 | 7.573e+05 | 7132 | 0.009271 | 0.009238 | 0.01311 |
| benchmarks_jm.csv | Reuters | T-MNB | 0.003 | 812.2 | 71.32 | 0.5341 | 0.004517 | 10.05 |
| benchmarks_jm.csv | Reuters | TCNB | 0.1 | 4.142e+04 | 2377 | 0.05647 | 0.0563 | 0.0899 |
| benchmarks_jm.csv | Reuters | TWCNB | 0.03 | 4.142e+04 | 713.2 | 0.01694 | 0.01689 | 0.02697 |
| final_results.csv | 20NG bydate | MNB | 0.01 | 1.079e+05 | 1016 | 0.01033 | 0.004638 | 0.01773 |
| final_results.csv | 20NG bydate | CNB | 0.1 | 2.051e+06 | 1.016e+04 | 0.004935 | 0.004837 | 0.00524 |
| final_results.csv | 20NG bydate | WCNB | 0.3 | 2.051e+06 | 3.049e+04 | 0.01481 | 0.01451 | 0.01572 |
| final_results.csv | 20NG bydate | T-MNB | 0.01 | 4042 | 1016 | 0.2649 | 0.1915 | 0.357 |
| final_results.csv | 20NG bydate | TCNB | 0.3 | 7.68e+04 | 3.049e+04 | 0.396 | 0.3909 | 0.4037 |
| final_results.csv | 20NG bydate | TWCNB | 0.3 | 7.68e+04 | 3.049e+04 | 0.396 | 0.3909 | 0.4037 |
| benchmarks_jm.csv | 20NG bydate | MNB | 0.03 | 1.079e+05 | 3049 | 0.03098 | 0.01392 | 0.05318 |
| benchmarks_jm.csv | 20NG bydate | CNB | 0.3 | 2.051e+06 | 3.049e+04 | 0.01481 | 0.01451 | 0.01572 |
| benchmarks_jm.csv | 20NG bydate | WCNB | 0.3 | 2.051e+06 | 3.049e+04 | 0.01481 | 0.01451 | 0.01572 |
| benchmarks_jm.csv | 20NG bydate | T-MNB | 0.03 | 4042 | 3049 | 0.7948 | 0.5746 | 1.071 |
| benchmarks_jm.csv | 20NG bydate | TCNB | 0.3 | 7.68e+04 | 3.049e+04 | 0.396 | 0.3909 | 0.4037 |
| benchmarks_jm.csv | 20NG bydate | TWCNB | 0.3 | 7.68e+04 | 3.049e+04 | 0.396 | 0.3909 | 0.4037 |

Spread across the six variants (std of log10, mean over splits; smaller = more agreement):

| source | setting | sd_log10_alpha | sd_log10_r | n_splits |
|---|---|---|---|---|
| final_results.csv | 20NG random | 0.8814 | 0.8020 | 10 |
| benchmarks_jm.csv | 20NG random | 0.8441 | 0.8837 | 10 |
| final_results.csv | Reuters | 0.6278 | 0.7868 | 1 |
| benchmarks_jm.csv | Reuters | 0.8394 | 0.6083 | 1 |
| final_results.csv | 20NG bydate | 0.6619 | 0.8046 | 1 |
| benchmarks_jm.csv | 20NG bydate | 0.4714 | 0.7256 | 1 |

## Null simulation (identical classes, raw counts)

Share of test documents assigned to the small class (mean ± std over seeds); the unbiased value is 1/(number of classes). In the two-class scenario the complement of one class is the other class, so CNB makes exactly the same decisions as MNB at the same smoothing; the five-class scenario (one small, four large classes) is added so that CNB differs.

### two-class (unbiased share 0.50)

| n_small | MNB a=1 | MNB a=0.01 | MNB-JM 0.1 | MNB-JM 0.01 | MNB-JM 0.001 | CNB a=1 | CNB-JM 0.1 |
|---|---|---|---|---|---|---|---|
| 5 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| 10 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| 20 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0001 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0001 |
| 50 | 0.0000 ± 0.0001 | 0.0001 ± 0.0002 | 0.0050 ± 0.0025 | 0.0000 ± 0.0001 | 0.0000 ± 0.0000 | 0.0000 ± 0.0001 | 0.0050 ± 0.0025 |
| 100 | 0.0038 ± 0.0014 | 0.0030 ± 0.0011 | 0.0523 ± 0.0067 | 0.0032 ± 0.0010 | 0.0004 ± 0.0004 | 0.0038 ± 0.0014 | 0.0523 ± 0.0067 |
| 500 | 0.5010 ± 0.0172 | 0.4999 ± 0.0269 | 0.5028 ± 0.0193 | 0.4999 ± 0.0198 | 0.4998 ± 0.0262 | 0.5010 ± 0.0172 | 0.5028 ± 0.0193 |

### five-class (unbiased share 0.20)

| n_small | MNB a=1 | MNB a=0.01 | MNB-JM 0.1 | MNB-JM 0.01 | MNB-JM 0.001 | CNB a=1 | CNB-JM 0.1 |
|---|---|---|---|---|---|---|---|
| 5 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0001 | 0.0000 ± 0.0000 |
| 10 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0003 ± 0.0003 | 0.0001 ± 0.0002 |
| 20 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0011 ± 0.0008 | 0.0011 ± 0.0007 |
| 50 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0016 ± 0.0009 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0063 ± 0.0015 | 0.0076 ± 0.0020 |
| 100 | 0.0004 ± 0.0005 | 0.0001 ± 0.0002 | 0.0166 ± 0.0034 | 0.0003 ± 0.0003 | 0.0000 ± 0.0000 | 0.0220 ± 0.0033 | 0.0258 ± 0.0032 |
| 500 | 0.1993 ± 0.0150 | 0.1993 ± 0.0150 | 0.2015 ± 0.0200 | 0.1996 ± 0.0144 | 0.1988 ± 0.0145 | 0.1970 ± 0.0114 | 0.1999 ± 0.0169 |

## Limitations

- Reuters and the chronological 20 Newsgroups split are single fixed splits; their rows carry no between-split variance, and per-split differences there are not tested.
- λ = 0.1 was fixed a priori; the λ sweep shows how much depends on it.
- JM interpolates toward the uniform distribution, not toward a corpus background model; a background model is the more common choice in language-model retrieval and could behave differently.
- The imbalance stress test shrinks half the classes of one dataset (20 Newsgroups); the recovery ratio is defined on that design only.
- The 20NG random splits overlap; the corrected t-test accounts for this only approximately.
- The committed results/final_results.csv was tuned with an earlier protocol for some tuned rows; this experiment re-tunes with the current src.experiments.pick_best.

## What this means for the paper's Sec. 3.1 claim

On the stress test at 10 %, replacing α = 1 by JM (λ = 0.1) in plain MNB recovers R = 1.22 of the gain that Complement NB brings (CI [1.03, 1.48]); by the pre-declared rule: R >= 0.8: smoothing imbalance explains most of the skewed-data bias. MNB-JM's shrunk-class recall is not significantly different (after Bonferroni) than CNB α = 1's (+3.41 points), and adding the complement on top of equal smoothing (CNB-JM vs MNB-JM) is not significantly different (after Bonferroni) (-4.56 points). On the 20 Newsgroups random splits the complement is worth +4.72 accuracy points at α = 1 but +1.30 points when both models use JM (significantly higher). The null simulation shows that the bias against a small class does not need unequal smoothing: with identical classes and JM, a class with 5 training documents receives at most 0.000 of the test documents (unbiased: 0.5). In that setting the estimation effect of Sec. 3.1 alone is enough to produce strong bias, so the two causes are not mutually exclusive. These results speak to one dataset design and one fixed λ; they do not show that the concavity argument is wrong, only how much of Complement NB's measured benefit on these benchmarks can also be obtained by equalising the smoothing.

