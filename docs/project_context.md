# Project context: House Price Prediction (ML CIA-2, Phase 2)

A factual summary of the project for help with the article's Abstract, Introduction and Literature Survey.
It lists **facts and numbers only**. Every number here was produced by code that was actually run. The article prose must be written by the team (the article is checked by Turnitin for plagiarism and AI writing).

Last updated: 20 November 2026, 17:15 (after the final notebook run; all results final).

---

## 1. Course and team

- **Course:** Machine Learning, Concept Integration Assessment 2 (Model Design and Implementation), BITS Pilani Dubai.
- **Team:** Nabhya Yadav (2024A7PS0615U, project lead), MVS Maheedhar Reddy (2024A7PS0624U), Nivedita Mahesh (2024A7PS0619U), Ponnamanda Hruday Manas (2024A7PS0609U).
- **Deliverables (due 20 Nov 2026, midnight):** article PDF (10–15 pages, Times New Roman 12 pt, 1.5 line spacing), code ZIP, slides. File names: `YourID_Article.pdf`, `YourID_Code.zip`, `YourID_Phase2_Slides.pptx`.
- **Article sections required by the brief:** (a) Abstract, (b) Introduction, (c) Literature Survey, (d) Implementation, (e) Architecture diagram, (f) Algorithm, (g) Results and Discussion, (h) Conclusion, (i) References (IEEE or APA, consistent with Phase 1).
- **Presentation:** 8–12 December (the brief says 2025; probably a typo for 2026; confirm with the instructor).

## 2. Phase 1 (already submitted)

- Literature review of **15 papers (2021–2025)** on house price prediction, plus problem definition. The paper list and summaries are in the team's Phase 1 report (not included here: upload or paste it separately).
- One Phase 1 paper relevant to our results: **Sharma et al. (2024)** also used the Ames dataset and found **XGBoost** performed best.
- Phase 1 contribution claimed: a **fair comparison** of models, i.e. every model gets a comparable tuning budget.

**Three problem statements from Phase 1:**
1. Build an accurate **and** interpretable gradient-boosting model, explained with **SHAP**.
2. Measure how much **low-cost location features** (neighbourhood encoding, neighbourhood price lag) improve prediction.
3. Test transfer to the **Dubai** market: **out of scope for this submission**, listed as future work.

## 3. Task and data

- **Task:** supervised **regression**: predict a house's sale price from its characteristics.
- **Dataset:** Ames Housing (Ames, Iowa, USA), Kaggle competition "House Prices: Advanced Regression Techniques"; compiled by De Cock (2011). Only Kaggle's `train.csv` is used (its `test.csv` has no prices): **1,460 houses × 81 columns** (79 features + `Id` + `SalePrice`).
- **Target:** `log(1 + SalePrice)`. SalePrice is right-skewed (skewness **1.88**); after the log transform the skewness is **0.12**. Predictions are converted back to dollars with `exp(x) − 1`.
- **Metrics:** RMSE on log price (also the Kaggle metric), MAE in USD, R², MAPE. Accuracy / precision / recall / F1 are **classification** metrics and do not apply to regression; the brief's "other relevant evaluation metrics" are used instead.
- **Environment:** Google Colab, CPU runtime, Python 3.13. Library versions used: pandas 2.2.3, numpy 2.4.6, scikit-learn 1.6.1, xgboost 3.4.1, shap 0.54.1. `random_state = 42` everywhere.

## 4. Method (what the code does, in order)

1. **Load:** drop `Id` (row number); treat `MSSubClass` (dwelling-type code) as categorical.
2. **Split first (prevents data leakage):** stratified 80/20 train/test split on 10 quantile bins of log price. Train **1,168** houses, test **292**. Price distributions of the two sets are almost identical (mean $180,977 vs $180,698; median $163,000 vs $163,495).
3. **Outliers (training set only):** houses with above-ground living area > 4,000 sq ft **and** price < $300,000. Two training houses were flagged (Kaggle Ids 524 and 1299, 4,676 and 5,642 sq ft, sold for $184,750 and $160,000), both **partial sales** (sold before construction was finished), as described by De Cock (2011). Two other large houses ($755,000 and $745,000) follow the trend and were kept. Training set after removal: **1,166** houses. The test set is not filtered.
4. **Preprocessing pipeline** (scikit-learn `Pipeline`, refitted inside every CV fold, so no statistic is learned from the houses being scored):
   - **Imputation** (custom transformer): "NA means absent" columns (pool, alley, fence, fireplace, garage and basement descriptors, masonry veneer type) → `"None"`; `MasVnrArea` → 0; `LotFrontage` (259 missing) → median of the same neighbourhood (training data); `Electrical` (1 missing) → mode; the one house with a basement but missing `BsmtExposure`/`BsmtFinType2` → mode; `GarageYrBlt` dropped (81 missing, near-duplicate of `YearBuilt`). 6,267 missing values filled in the training set; 0 remain in train or test.
   - **Ordinal encoding** of 10 quality/condition columns: None/Po/Fa/TA/Gd/Ex → 0–5.
   - **Feature engineering** (5 features): `TotalSF` = basement + 1st + 2nd floor area; `TotalBath` = full + 0.5·half baths (incl. basement); `HouseAge` = year sold − year built; `YearsSinceRemod` = year sold − year remodelled; `QualxSize` = overall quality × living area.
   - **One-hot encoding** of the remaining categorical columns (unknown categories → all zeros) and **standard scaling** of numeric columns.
   - Result: **277** model input columns (44 numeric + 5 engineered + 228 one-hot).
5. **Models, simple → complex,** each evaluated with the **same 5-fold cross-validation** on the training set (this is the validation step), with a common tuning budget of **30 candidate settings per tuned model**: Linear Regression (baseline, no tuning), Ridge (grid over alpha), Lasso (grid over alpha), Random Forest (randomized search), XGBoost (randomized search).
6. **Model selection:** lowest mean CV RMSE; the selected model is refitted on all 1,166 training houses and evaluated **once** on the 292 test houses.
7. **Bias–variance analysis:** train vs CV RMSE for every model; learning curve for the best model; validation curve for Ridge alpha.
8. **Location ablation (problem statement 2):** best model with Neighborhood (a) removed, (b) one-hot encoded, (c) target-encoded (Tier 1, scikit-learn `TargetEncoder`, cross-fitted to avoid leakage). Tier 2 (neighbourhood price lag) **not implemented**.
9. **Interpretation:** SHAP `TreeExplainer` on the tuned XGBoost model (summary/beeswarm plot + one waterfall plot).

## 5. Results (5-fold CV on the training set, RMSE of log price; lower is better)

| Model | Best hyperparameters | Train RMSE | CV RMSE (mean ± std) | Gap (CV − train) |
|---|---|---|---|---|
| Linear Regression | none | 0.0858 | 0.1324 ± 0.0100 | 0.0466 |
| Ridge | alpha ≈ 18.87 | 0.0978 | 0.1148 ± 0.0131 | 0.0170 |
| **Lasso** | alpha ≈ 0.000672 | 0.0992 | **0.1123 ± 0.0122** | 0.0131 |
| Random Forest (1st run, all features per split) | 100 trees, min leaf 2, max depth 15 | 0.0602 | 0.1330 ± 0.0124 | 0.0728 |
| Random Forest (re-tuned incl. max_features) | 300 trees, min leaf 1, max_features 0.33, max depth 30 | 0.0487 | 0.1312 ± 0.0131 | 0.0825 |
| XGBoost (1st run) | 1000 trees, learning rate 0.05, depth 3, subsample 0.7, colsample 0.3 | 0.0313 | 0.1161 ± 0.0106 | 0.0848 |
| **XGBoost (re-run with wider ranges, final)** | 1500 trees, learning rate 0.03, depth 3, subsample 0.8, colsample 0.1 | 0.0408 | 0.1156 ± 0.0093 | 0.0748 |

Observations so far (from the numbers above):
- Regularisation helps: Ridge and Lasso raise the training error but lower the CV error compared with plain Linear Regression, and better on all 5 folds (bias–variance trade-off).
- Lasso keeps **92 of 277** features (185 coefficients exactly zero) with the best CV RMSE so far. All 5 engineered features were kept. `TotalSF` has the largest numeric coefficient; three neighbourhoods (Crawfor, StoneBr, BrkSide) are among the 10 largest coefficients.
- The tree ensembles fit the training data much more closely (large train–CV gap). XGBoost (boosting) beat Random Forest (bagging) on all 5 folds, but in its first run did not beat Lasso.
- XGBoost's first run had its best values at the edges of two search ranges, so its ranges were widened and the search re-run (result pending).

- XGBoost re-run with wider ranges: 0.1156 (first run 0.1161), better on 4 of 5 folds than its first run, still behind Lasso (worse on 4 of 5 folds). `colsample_bytree` = 0.1 was again at the edge of its range; not widened further (deadline) → limitation.

**Final ranking (CV RMSE):** Lasso 0.1123 < Ridge 0.1148 < XGBoost 0.1156 < Random Forest 0.1312 < Linear Regression 0.1324. The top three differ by less than one fold standard deviation (~0.01). Lasso beats Ridge on all 5 folds.

## 5b. Final test-set evaluation (selected model: Lasso, 292 test houses, used once)

| Metric | Value |
|---|---|
| RMSE (log price) | 0.1096 (CV estimate 0.1123) |
| MAE | $14,235 |
| R² (log price / USD) | 0.9284 / 0.9307 |
| MAPE | 8.22% |

- Test RMSE is close to (slightly better than) the CV estimate → the CV estimate was reliable.
- Largest errors: four very cheap houses over-predicted by about 43–54% (actual $35,311 / $52,000 / $39,300 / $66,500 vs predicted $54,191 / $79,479 / $57,900 / $95,299); one $159,434 house predicted at $112,967. Two of the three most expensive test houses (about $500k and $610k) are predicted well below their price (about $440k and $475k, read from the plot); the most expensive (about $625k) almost exactly.
- Residuals centred on 0, wider spread at low predicted prices.

## 5c. Bias–variance

- Gaps (CV − train): Lasso 0.013, Ridge 0.017, Linear 0.047, XGBoost 0.075, Random Forest 0.082 → regularised linear models low variance; tree ensembles high variance (very low training error).
- Learning curve (Lasso), training houses → train / CV RMSE: 186 → 0.0592 / 0.2623; 372 → 0.0833 / 0.1250; 559 → 0.0919 / 0.1176; 745 → 0.0983 / 0.1133; 932 → 0.0992 / 0.1123. Curves nearly converge; CV still falling slightly → more data would help a little; remaining error mostly bias.
- Ridge validation curve: U-shaped CV error with minimum at alpha ≈ 18.9; small alpha = overfitting (low train, higher CV error), large alpha (1,000) = underfitting (both high).

## 5d. Location ablation (problem statement 2; Lasso, same tuned alpha, 5-fold CV)

| Version | CV RMSE | Change vs no location |
|---|---|---|
| (a) no location (Neighborhood dropped) | 0.1155 ± 0.0123 | — |
| (b) Neighborhood one-hot | 0.1123 ± 0.0122 | −0.0032 (better on all 5 folds) |
| (c) Neighborhood target encoding (Tier 1) | 0.1137 ± 0.0129 | −0.0017 (better on 4 of 5 folds) |

- Location adds a consistent but small improvement (~3% relative), smaller than the fold spread. Likely because quality, size, age, zoning and lot features already carry much of the location signal (Neighborhood is also used to impute LotFrontage in all versions). Target encoding (1 column) recovers about half the one-hot gain. Tier 2 price lag not implemented.

## 5e. SHAP (XGBoost, test set)

- Top features by mean |SHAP| (log price units): QualxSize 0.044, OverallQual 0.043, TotalSF 0.029, GrLivArea 0.028, BsmtQual 0.023, TotalBath 0.022, OverallCond 0.019, KitchenQual 0.014, Fireplaces 0.014, YearBuilt 0.014, ExterQual 0.014, LotArea 0.013, YearRemodAdd 0.012, GarageArea 0.012, 1stFlrSF 0.012.
- 3 of the top 6 are engineered features (QualxSize, TotalSF, TotalBath). High quality/size/bathrooms/fireplaces/build year → higher predicted price.
- No Neighborhood column in XGBoost's top 15 (consistent with the small ablation effect).
- Waterfall for one typical test house (Kaggle Id 630): actual $168,500, predicted $163,559; prediction 12.005 vs model average 12.026 (log scale); small positive and negative contributions nearly cancel.
- SHAP shows what the model learned (association), not causation.

## 6. Points for the Introduction / motivation (facts only)

- House prices matter to buyers, sellers, lenders and planners; automated valuation needs accuracy **and** explanations people can trust (hence SHAP).
- Location is known to drive price, but rich location data (coordinates, amenities) is costly; problem statement 2 tests how much cheap location features (neighbourhood identity) add.
- The Ames dataset is a standard benchmark with 79 descriptive features, which makes results comparable with published work (e.g. Sharma et al., 2024).

## 7. Limitations already known

- Single, small dataset (1,460 houses) from one US city, sold 2006–2010 (per De Cock, 2011; not re-checked in our code); no Dubai data (problem statement 3 left as future work).
- No geographic coordinates, so no true spatial-lag features; Tier 2 neighbourhood price lag not implemented.
- Hyperparameter search used a limited budget (30 candidates per model) on CPU; XGBoost's best `colsample_bytree` (0.1) was at the edge of its range.
- Ablation used Lasso's hyperparameters tuned with one-hot location for all three versions (not re-tuned per version).
- Comparison point with Phase 1: Sharma et al. (2024) found XGBoost best on Ames; in our setup (log target, engineered features, outlier removal, equal tuning budgets) regularised Lasso was slightly better than XGBoost, though the top three models are within one fold standard deviation. Differences in preprocessing, tuning and evaluation could explain this; it was not investigated further.

## 8. Files the article can use (after the run)

- Figures in `outputs/figures/`: price and log-price histograms (EDA), living area vs price (training set), model comparison, learning curve, Ridge validation curve, predicted vs actual (test), residuals (test), SHAP summary, SHAP waterfall, location ablation, architecture diagram.
- Tables in `outputs/results/`: `cv_results.csv`, `test_metrics.csv`, `ablation.csv`.
