# Fixing Naive Bayes for Text Classification

A from-scratch reimplementation and re-evaluation of Rennie et al. (2003),
*Tackling the Poor Assumptions of Naive Bayes Text Classifiers* (ICML).
Course project, AI & Machine Learning, Digital University Kerala.

**Team:** Adhil Rifayin K S, Suryakanth T S · **Blog:** [link] · **Report:** [link]

## What we did
We built the paper's classifier one correction at a time (MNB → CNB → WCNB →
transforms → TCNB → TWCNB), verified every variant against scikit-learn, and
compared all six with a linear SVM on 20 Newsgroups and Reuters-21578, under
the paper's settings (α = 1) and with smoothing tuned on validation data.

## Results
Accuracy for 20 Newsgroups (mean of 10 random splits) and the chronological split;
macro-F1 for Reuters.

| Model | 20NG random, paper | 20NG random, tuned | Reuters, paper | Reuters, tuned | 20NG chronological, tuned |
|---|---|---|---|---|---|
| MNB | 0.854 | 0.892 | 0.233 | 0.593 | 0.646 |
| CNB | 0.901 | 0.903 | 0.529 | 0.519 | 0.667 |
| WCNB | 0.879 | 0.879 | 0.517 | 0.604 | 0.662 |
| T-MNB | 0.883 | 0.920 | 0.149 | 0.609 | 0.695 |
| TCNB | 0.913 | 0.914 | 0.428 | 0.538 | 0.718 |
| TWCNB | 0.910 | 0.913 | 0.326 | 0.586 | 0.716 |
| LinearSVC | 0.939 | 0.939 | 0.768 | 0.743 | 0.706 |

## Key findings
- **The paper's diagnosis holds:** at its settings, the corrections lift 20 Newsgroups
  accuracy from 0.854 to 0.913, and the complement rescues small Reuters classes.
- **Smoothing is a hidden variable:** with α tuned per model, plain MNB beats CNB on
  Reuters, and weight normalization never gives a consistent gain.
- **The SVM's lead depends on the setting:** clearly ahead on random splits, matched by
  corrected Naive Bayes on the chronological split, while training 15–70× slower.
- **Table 4, step 6 has a sign error:** implemented literally, accuracy drops to 1.9%.

## Repository
- `src/twcnb.py` – the classifier from scratch (NumPy/SciPy); every correction switchable
- `src/data.py` – 20 Newsgroups and single-label Reuters loaders
- `src/experiments.py` – final comparison, paper and tuned protocols
- `src/significance.py` – paired t-tests (plain and Nadeau–Bengio corrected)
- `src/grid_check.py` – re-tuning on a wider smoothing grid
- `src/imbalance.py` – stress test: shrinking half the classes
- `src/coin_simulation.py` – exact reproduction of the paper's Table 1
- `src/plots.py` – figures
- `tests/` – equivalence checks against scikit-learn
- `notebooks/build_step_by_step.ipynb` – the step-by-step build with every check

## Reproduce
```bash
pip install -r requirements.txt
pytest -v
python -m src.experiments        # writes results/final_results.csv (slow: SVM)
python -m src.significance
python -m src.grid_check
python -m src.imbalance
python -m src.coin_simulation
python -m src.plots
```

## Reference
Rennie, J. D. M., Shih, L., Teevan, J., & Karger, D. R. (2003). Tackling the poor
assumptions of naive Bayes text classifiers. *Proceedings of ICML*.