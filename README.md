# House Price Prediction with Machine Learning (Ames Housing)

Machine Learning, Concept Integration Assessment 2 (Model Design and Implementation), BITS Pilani Dubai.

We predict house sale prices (supervised **regression**) on the Ames Housing data. We compare five models, from linear regression to XGBoost, under one leakage-free pipeline with equal tuning budgets, measure how much neighbourhood information adds, and explain the tree model with SHAP.

**Team:** Nabhya Yadav (2024A7PS0615U, lead) · MVS Maheedhar Reddy (2024A7PS0624U) · Nivedita Mahesh (2024A7PS0619U) · Ponnamanda Hruday Manas (2024A7PS0609U)

## Results at a glance

5-fold cross-validation on the training set (RMSE of log sale price, lower is better):

| Model | CV RMSE (mean ± std) | Train RMSE |
|---|---|---|
| **Lasso** (selected) | **0.1123 ± 0.0122** | 0.0992 |
| Ridge | 0.1148 ± 0.0131 | 0.0978 |
| XGBoost | 0.1156 ± 0.0093 | 0.0408 |
| Random Forest | 0.1312 ± 0.0131 | 0.0487 |
| Linear Regression | 0.1324 ± 0.0100 | 0.0858 |

Selected model (Lasso) on the held-out test set (292 houses, used once): **RMSE (log) 0.1096 · MAE $14,235 · R² 0.93 · MAPE 8.2%**.

Full tables: `outputs/results/cv_results.csv`, `test_metrics.csv`, `ablation.csv`.

## Repository layout

```
├── README.md
├── requirements.txt
├── notebooks/
│   ├── 01_eda.ipynb          # exploratory data analysis
│   └── 02_modelling.ipynb    # full pipeline: preprocessing → models → evaluation → SHAP
├── outputs/
│   ├── algorithm.md          # stepwise pseudocode of the pipeline
│   ├── figures/              # all figures (PNG, 150 dpi)
│   └── results/              # CSV tables
├── docs/project_context.md   # project summary used while writing the article
└── data/                     # put train.csv here (not included, see below)
```

## Getting the data

The data is Kaggle's **"House Prices: Advanced Regression Techniques"** competition: <https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data>.

- Download `train.csv` (1,460 houses × 81 columns) and place it at `data/train.csv`.
- Only `train.csv` is used. Kaggle's `test.csv` has no prices, so it cannot be used for evaluation.
- The CSV is **not included** in this repository, because Kaggle's competition rules restrict redistribution.
- In Colab, the notebook's setup cell downloads the team's shared Google Drive copy with `gdown` automatically. If that link is unavailable, upload `train.csv` to `data/` yourself.

## How to run

### Google Colab (recommended; this is how the results were produced)
1. Open `notebooks/02_modelling.ipynb` in Colab (CPU runtime is enough).
2. **Runtime → Run all.** The first cell installs the libraries; the setup cell downloads `train.csv` into `data/`.
3. Figures are written to `outputs/figures/` and tables to `outputs/results/`. Colab deletes these when the session ends, so download them before closing. The last cell displays every figure and prints every table, so copies are also kept inside the saved notebook.

### Locally
```bash
pip install -r requirements.txt
jupyter notebook notebooks/02_modelling.ipynb
```
Then run all cells. Paths in the notebook (`data/`, `outputs/`) are relative to the **working directory**, which in Jupyter is the notebook's own folder (`notebooks/`). The setup cell creates `notebooks/data/`, `notebooks/outputs/figures/` and `notebooks/outputs/results/`, and downloads `train.csv` there. If the download link is unavailable, put `train.csv` in `notebooks/data/` yourself.

## Dependencies

See `requirements.txt` (minimum versions). Results were produced in Google Colab, CPU runtime, Python 3.13, with:

| Library | Version |
|---|---|
| pandas | 2.2.3 |
| numpy | 2.4.6 |
| scikit-learn | 1.6.1 (≥ 1.4 required for `TargetEncoder`) |
| xgboost | 3.4.1 |
| shap | 0.54.1 |
| matplotlib, gdown | as installed in Colab |

All random steps use `random_state = 42`, so a re-run gives the same numbers.

## Expected outputs and run time

On a Colab CPU runtime, the timed hyperparameter searches took about 17 minutes together (Random Forest 7.2 min, XGBoost 8.7 min, Ridge about 0.7 min). Everything else is quicker. The total run time was not measured separately.

**Figures** (`outputs/figures/`):

| File | Shows |
|---|---|
| `price_hist.png`, `log_price_hist.png` | sale price before and after the log transform (EDA notebook) |
| `train_grlivarea_vs_saleprice.png` | living area vs price, with the two removed outliers |
| `model_comparison_cv_rmse.png` | train vs CV RMSE for all five models |
| `learning_curve_lasso.png` | train / CV RMSE vs number of training houses |
| `validation_curve_ridge_alpha.png` | train / CV RMSE across Ridge's regularisation strength |
| `predicted_vs_actual_test.png`, `residuals_test.png` | test-set predictions and errors |
| `ablation_location_cv_rmse.png` | CV RMSE without location / one-hot / target-encoded neighbourhood |
| `shap_summary_xgboost.png`, `shap_waterfall_xgboost.png` | SHAP explanations of the XGBoost model |
| `architecture.png` | workflow diagram of the pipeline |

**Tables** (`outputs/results/`): `cv_results.csv` (train RMSE, CV RMSE mean ± std, gap, best hyperparameters per model), `test_metrics.csv` (test metrics of the selected model), `ablation.csv` (location ablation).

## Method in brief

1. Drop `Id`; treat `MSSubClass` as categorical; target = log(1 + SalePrice) (skewness 1.88 → 0.12).
2. **Split first:** stratified 80/20 train/test split (1,168 / 292 houses), so no statistic is ever learned from the test set.
3. Remove 2 partial-sale outliers (> 4,000 sq ft, < $300,000) from the **training set only**.
4. One scikit-learn `Pipeline`, refitted inside every CV fold: rule-based imputation (neighbourhood median for `LotFrontage`), ordinal quality grades (0–5), 5 engineered features (`TotalSF`, `TotalBath`, `HouseAge`, `YearsSinceRemod`, `QualxSize`), one-hot encoding and standard scaling.
5. Five models tuned with the **same 5 folds** and **30 candidate settings each** (grid search for Ridge/Lasso, randomized search for Random Forest/XGBoost).
6. Select the lowest CV RMSE, refit on the full training set, evaluate **once** on the test set.
7. Bias–variance analysis (train vs CV error, learning curve, validation curve), location ablation, SHAP.

Pseudocode: `outputs/algorithm.md`.

## Why regression metrics, not accuracy / precision / recall / F1

The brief lists accuracy, precision, recall and F1-score, but these are **classification** metrics: they count correct and incorrect class labels. House-price prediction is **regression** (the output is a continuous price), so we use the brief's "other relevant evaluation metrics": **RMSE** on log price (Kaggle's metric; roughly a relative error), **MAE** in dollars, **R²**, and **MAPE**.

## Limitations and future work

- One small dataset (1,460 houses) from a single US city.
- No geographic coordinates, so no true spatial features. The neighbourhood price-lag feature (Tier 2) is not implemented.
- Tuning was limited to 30 candidates per model on CPU. XGBoost's best `colsample_bytree` sat at the edge of its search range.
- **Transfer to the Dubai housing market** (Phase 1 problem statement 3) is future work: it needs a comparable Dubai dataset.
