# CLAUDE.md — House Price Prediction (ML Course Project, BITS Pilani Dubai)

This file tells Claude Code how to work in this repository. Read it fully before doing anything.

## 1. Project context

- **Course:** Machine Learning — Concept Integration Assessment 2 (Model Design and Implementation).
- **Topic:** House price prediction using machine learning (supervised **regression**).
- **Team:** Nabhya Yadav (2024A7PS0615U, project lead), MVS Maheedhar Reddy (2024A7PS0624U), Nivedita Mahesh (2024A7PS0619U), Ponnamanda Hruday Manas (2024A7PS0609U).
- **Phase 1 (done):** literature review of 15 papers (2021–2025) and problem definition. Its three problem statements drive this code:
  1. Build an accurate **and** interpretable gradient boosting model (explained with SHAP).
  2. Measure how much low-cost location features (neighbourhood encoding, neighbourhood price lag) improve prediction.
  3. Test transfer to the Dubai market — **out of scope for this submission; it goes in "Future work".**
- **Phase 2 deadline:** 20 November 2026 (extended). Deliverables uploaded to LMS: article PDF (10–15 pages), code ZIP, slides.
- **File names for submission:** `YourID_Article.pdf`, `YourID_Code.zip`, `YourID_Phase2_Slides.pptx`.

## 2. How to work with Nabhya (read carefully)

### Git: commits are Nabhya's job, never yours
- **Never run `git commit`, `git push`, `git tag`, `git rebase`, `git reset`, `git merge` or `git stash`**, in any form (including `git -C`, `sh -c`, aliases or scripts). `.claude/settings.json` blocks the common forms, but treat this as an absolute rule regardless.
- You may run read-only git commands (`git status`, `git diff`, `git log`) to see what changed.
- **Do not run `git add` either.** Nabhya stages and commits herself after reviewing your changes.
- When a unit of work is finished, stop and tell her:
  1. Which files you changed and why (2–4 bullet points).
  2. How to verify it (what to run, what output to expect).
  3. A **suggested** commit message she can use or rewrite, e.g. `feat: add preprocessing pipeline with neighbourhood-aware imputation`.
- Then wait. Do not start the next step until she says she has reviewed (and usually committed) the current one.

### Teaching style
- Nabhya is new to ML/DL terminology. **Every time a new ML term appears, define it in one plain sentence**, and end each explanation with a short **Glossary** section.
- Explain **why** a step is done, not just what the code does. She must be able to defend every line in a live presentation to the instructor (8–12 December).
- Work in **small steps**: one function, one cell or one file per step. No giant one-shot rewrites.
- If she is wrong about something, say so directly and explain why. Be rigorous; she checks reasoning for gaps.
- Keep answers focused; she asks follow-up questions.

### Honesty about results
- **Never invent, estimate or "typical-value" any metric, number or chart description.** All numbers in the README, slides notes or article come only from code that was actually run, with the output pasted back or saved to `outputs/`.
- If something hasn't been run, say "not yet run".

### Article and slides
- The research article is checked by Turnitin (plagiarism and AI-writing detection). **Do not write article prose for submission.** You may: give outlines, list the facts/numbers to include, explain results, check her text for technical accuracy, and generate figures/tables from code.
- The README and code comments are part of the code deliverable; you may write those.

## 3. Environment

- Code runs in **Google Colab (Python 3.13), CPU runtime**, usually via the Colab extension in VS Code.
- Nabhya's laptop has Python 3.14, where some libraries (e.g. SHAP) may not install cleanly. **Do not try to install packages locally or debug local environment problems.** If you need something run, ask her to run it in Colab and paste the output.
- Colab sessions are temporary: files in `/content/` are deleted when the session ends. Notebooks must start with a setup cell that installs/imports everything and downloads the data, so "Run All" rebuilds the session.
- Libraries: pandas, numpy, scikit-learn (>= 1.4, for `TargetEncoder`), xgboost, shap, matplotlib, seaborn (optional). Pin minimum versions in `requirements.txt`.
- Always set `random_state=42` (splits, CV, models) so results are reproducible.

## 4. Dataset

- **Ames Housing**, Kaggle competition "House Prices: Advanced Regression Techniques". Use **`train.csv` only** (1,460 rows × 81 columns). Kaggle's `test.csv` has no prices, so it cannot be used for evaluation.
- Do **not** commit the CSV (Kaggle's competition rules restrict redistribution). `data/` is git-ignored; the README explains how to download it.
- Colab download: Nabhya keeps a shared Drive copy and downloads it with `!gdown <file-id>` into `data/train.csv`.

## 5. Decisions already made in EDA (do not change without asking)

| Topic | Decision | Reason |
|---|---|---|
| Target | Model `np.log1p(SalePrice)`; convert predictions back with `np.expm1` | Skew 1.88 → 0.12 after log |
| `Id` | Drop | Row number, no information |
| `MSSubClass` | Treat as **categorical** (cast to string) | It is a dwelling-type code, not a quantity |
| `GarageYrBlt` | Drop | 81 missing; near-duplicate of `YearBuilt` |
| "NA means absent" columns: `PoolQC, MiscFeature, Alley, Fence, FireplaceQu, MasVnrType, GarageType, GarageFinish, GarageQual, GarageCond, BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1, BsmtFinType2` | Fill with the string `"None"` | Missing = house has no pool/garage/basement etc. |
| `MasVnrArea` | Fill with 0 | No veneer = 0 area |
| `LotFrontage` (259 missing) | Fill with the **median of the same Neighborhood**, computed on training data only | Truly unknown; lot widths are similar within a neighbourhood |
| `Electrical` (1 missing) | Fill with mode (training data) | Genuinely unknown |
| The 1 house with a basement but missing `BsmtExposure`/`BsmtFinType2` | Fill with mode, not "None" | Counts are 38 vs 37 for other basement columns |
| Outliers | Remove houses with `GrLivArea > 4000` and `SalePrice < 300000` **from the training set only** (confirm with Nabhya after the scatter plot) | Known partial/unusual sales in Ames |
| Multicollinearity | Note pairs (GarageCars/GarageArea, TotalBsmtSF/1stFlrSF, GrLivArea/TotRmsAbvGrd); keep them, let ridge/lasso handle it | Discussed in article |

Quality columns (`ExterQual, ExterCond, BsmtQual, BsmtCond, HeatingQC, KitchenQual, FireplaceQu, GarageQual, GarageCond, PoolQC`) are **ordinal**: map `None/Po/Fa/TA/Gd/Ex` → `0/1/2/3/4/5`. Other categoricals: one-hot encoding (`handle_unknown="ignore"`).

## 6. The pipeline to build (in this order)

**Golden rule: no data leakage.** Split first. Every statistic (medians, modes, encodings, scaling, lag features) is learned from training data only, inside a scikit-learn `Pipeline`/`ColumnTransformer` or a custom transformer with `fit`/`transform`.

1. **Load** `data/train.csv`, drop `Id`, cast `MSSubClass` to string.
2. **Split:** 80% train / 20% test with `train_test_split(random_state=42)`. Stratify on binned log price (e.g. 10 quantile bins) so both sets have similar price ranges. The test set is untouched until step 8.
3. **Remove outliers** from the training set only (Section 5).
4. **Preprocessing** (custom transformer + `ColumnTransformer`): imputation rules from Section 5, ordinal mapping, one-hot encoding, `StandardScaler` on numeric features (needed for linear/ridge/lasso; harmless for trees).
5. **Feature engineering:** `TotalSF = TotalBsmtSF + 1stFlrSF + 2ndFlrSF`; `TotalBath = FullBath + 0.5*HalfBath + BsmtFullBath + 0.5*BsmtHalfBath`; `HouseAge = YrSold - YearBuilt`; `YearsSinceRemod = YrSold - YearRemodAdd`; `QualxSize = OverallQual * GrLivArea`.
6. **Location features** (for the ablation):
   - Tier 1: `Neighborhood` target encoding with `sklearn.preprocessing.TargetEncoder` (it cross-fits internally, which prevents leakage).
   - Tier 2 (if time allows): neighbourhood price lag = mean log price of **training** sales in the same neighbourhood sold **before** this house's sale date (`YrSold`, `MoSold`). Fall back to the training neighbourhood mean when there are no earlier sales.
7. **Models, simple → complex**, each tuned with **5-fold CV on the training set** (this is the "validation" split the brief requires):
   1. Linear Regression (baseline)
   2. Ridge (tune `alpha`)
   3. Lasso (tune `alpha`; report how many features it zeroes out)
   4. Random Forest (`n_estimators`, `max_depth`, `min_samples_leaf`)
   5. XGBoost (`n_estimators`, `learning_rate`, `max_depth`, `subsample`, `colsample_bytree`) — `RandomizedSearchCV`, modest budget so it runs in minutes on CPU
   6. Stacking (if time allows): `StackingRegressor` of Ridge + RF + XGBoost with a Ridge meta-model
   Give every model a comparable tuning budget (fair comparison is a Phase 1 contribution).
8. **Bias–variance analysis:** for each model report train RMSE vs CV RMSE (underfitting = both high; overfitting = big gap). Plot a learning curve (`learning_curve`) for the best model and a validation curve for Ridge `alpha` (regularization).
9. **Model selection and final test:** pick the best model by CV RMSE, refit on the full training set, evaluate **once** on the test set.
10. **Metrics** (regression, so no accuracy/precision/recall/F1 — state this in the README):
    - RMSE on log price (Kaggle's metric), MAE in dollars, R², MAPE.
11. **Ablation** (problem statement 2): best model with (a) structural features only, (b) + Tier 1, (c) + Tier 2. Report CV RMSE for each.
12. **Interpretation:** SHAP (`TreeExplainer`) for the tree model — summary/beeswarm plot (global) and one waterfall plot (single house).

### Figures to save in `outputs/figures/` (150 dpi, titled, axis labels with units)
price histogram, log-price histogram, GrLivArea vs SalePrice scatter, missing-values bar chart, top-correlations bar chart, model comparison bar chart (CV RMSE), learning curve, validation curve (Ridge alpha), predicted vs actual (test set), residual plot, SHAP summary, SHAP waterfall, ablation bar chart.

### Tables to save in `outputs/results/` (CSV)
`cv_results.csv` (per model: train RMSE, CV RMSE mean ± std, best params), `test_metrics.csv`, `ablation.csv`.

### Priority if time runs out
Must have: steps 1–5, models 1–5, step 8 (train vs CV + one learning curve), steps 9–10, SHAP summary.
Nice to have: Tier 2 lag, stacking, validation curve, SHAP waterfall, full ablation.

## 7. Repository structure

```
house-price-prediction/
├── CLAUDE.md
├── README.md                 # how to run, dependencies, expected outputs
├── requirements.txt
├── .gitignore
├── .claude/settings.json     # blocks git commit/push for Claude
├── data/                     # train.csv goes here (git-ignored)
├── notebooks/
│   ├── 01_eda.ipynb          # Nabhya's existing exploration work
│   └── 02_modelling.ipynb    # runs the full pipeline, calls src/
├── src/                      # OPTIONAL: only if Nabhya asks (see below)
│   ├── data.py               # load, drop Id, split
│   ├── preprocessing.py      # imputation, encoding, scaling transformers
│   ├── features.py           # engineered + location features
│   ├── models.py             # model definitions and search spaces
│   └── evaluate.py           # metrics, CV, learning curves, plots
└── outputs/
    ├── figures/
    └── results/
```

**Default for this deadline: put the whole pipeline in `notebooks/02_modelling.ipynb`**, organised into clearly headed sections (markdown cells) that mirror the `src/` modules above. Reason: the notebook runs on a Colab machine, while the repo lives on Nabhya's laptop, so `import src...` would fail in Colab unless the repo is cloned there. Only split code into `src/` if Nabhya asks and the repo is available in Colab (e.g. `git clone` from GitHub in the setup cell).

## 8. Code standards

- Clear comments that explain **why** (the brief grades this). No commented-out dead code.
- Each function has a short docstring: what it does, inputs, outputs.
- Small functions; descriptive names; no magic numbers without a comment.
- Printed output is labelled and rounded, e.g. `print("CV RMSE (log):", round(score, 4))`.
- Every chart has a title, axis labels and units, and is saved before `plt.show()` with a **unique** filename.

## 9. README.md must contain

Project summary; team; how to get the data (Kaggle link + place in `data/train.csv`); how to run (Colab steps and local steps); dependencies (`pip install -r requirements.txt`); expected outputs (list of figures/CSVs and roughly how long it takes); note on why regression metrics are used instead of accuracy/precision/recall/F1; note that Dubai transfer is future work.

## 10. Notebook documentation (graded: "Explain steps taken for pre-processing, training, and evaluation")

Before every code section in the notebook, add a short markdown cell: **what** this step does, **why** it is needed, and **what to look for** in the output. After key outputs, add a markdown cell noting what the result shows (filled in only after the code has actually run).

## 11. What the brief asks for, and who does it

| Brief item | What Claude Code does | What the team does |
|---|---|---|
| 1.1(a–c) Abstract, Introduction, Literature Survey | Nothing to write; may check facts | Condense the Phase 1 report (rephrase; Turnitin compares against earlier submissions) |
| 1.1(d) Implementation | Lists the exact preprocessing steps, libraries, versions and environment used (from the code) | Writes the section |
| 1.1(e) Architecture diagram | Generates a workflow diagram as an image from the real pipeline (e.g. matplotlib/graphviz/Mermaid) → `outputs/figures/architecture.png`: data → split → preprocessing → features → models (CV tuning) → model selection → test evaluation → SHAP | Reviews it, may redraw in draw.io |
| 1.1(f) Algorithm | Produces stepwise pseudocode of the implemented pipeline (train/validation/test logic) in `outputs/algorithm.md`, matching the code exactly | Adapts it into the article and explains the logic in their own words |
| 1.1(g) Results and Discussion | Saves all metrics tables and figures; explains what they mean; lists comparison points with the Phase 1 papers (e.g. Sharma et al. 2024 also used Ames and found XGBoost best) | Writes the discussion |
| 1.1(h) Conclusion | Lists limitations observed in the code/results (e.g. single small US dataset, no Dubai data, no coordinates for true spatial lag) | Writes the section |
| 1.1(i) References | — | IEEE or APA, consistent with Phase 1 |
| 1.2 Code ZIP | Notebook(s), `requirements.txt`, README (how to run, dependencies, expected outputs, dataset download instructions) | Reviews, commits, zips |
| 1.4 Slides | Suggests a slide outline and which saved figures go on which slide | Builds the slides: problem statement, architecture + algorithm, results + discussion, conclusion |

**Metrics note for README and article:** the brief lists accuracy, precision, recall and F1-score, but those are **classification** metrics. This is a **regression** problem, so we use the brief's "other relevant evaluation metrics": RMSE (log price), MAE (USD), R² and MAPE. State this explicitly.

**Article formatting** (Phase 1 guidelines carry over): 10–15 pages, Times New Roman, 12 pt, 1.5 line spacing.

**Presentation:** 8–12 December (the brief says 2025; almost certainly a typo for 2026 — confirm with the instructor).

## 12. Submission checklist

- [ ] Preprocessing: missing values, standardization, feature engineering
- [ ] Algorithms chosen and justified
- [ ] Train / validation (CV) / test split
- [ ] Regression metrics + graphs + tables, with the note on why not accuracy/F1
- [ ] Bias–variance, regularization, model selection discussed
- [ ] Architecture diagram and algorithm pseudocode generated from the real pipeline
- [ ] Clear code comments + markdown explanations in the notebook
- [ ] README: how to run, dependencies, expected outputs, dataset download
- [ ] Files named `YourID_Article.pdf`, `YourID_Code.zip`, `YourID_Phase2_Slides.pptx`
- [ ] Code ZIP excludes `data/`, `.git/`, caches
