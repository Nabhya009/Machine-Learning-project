# Project context: House Price Prediction (ML CIA-2, Phase 2)

A factual summary of the whole project, for help with writing the article and slides.
It lists **facts and numbers only**. Every number was produced by code that was actually run (`notebooks/02_modelling.ipynb`; tables in `outputs/results/`). The article prose must be written by the team, because the article is checked by Turnitin for plagiarism and AI writing.

Last updated: 20 November 2026, 17:25. **All results are final.**

---

## 1. Course and team

- **Course:** Machine Learning, Concept Integration Assessment 2 (Model Design and Implementation), BITS Pilani Dubai.
- **Team:** Nabhya Yadav (2024A7PS0615U, project lead), MVS Maheedhar Reddy (2024A7PS0624U), Nivedita Mahesh (2024A7PS0619U), Ponnamanda Hruday Manas (2024A7PS0609U).
- **Deliverables (due 20 Nov 2026, midnight):** article PDF (10–15 pages, Times New Roman 12 pt, 1.5 line spacing), code ZIP, slides. File names: `YourID_Article.pdf`, `YourID_Code.zip`, `YourID_Phase2_Slides.pptx`.
- **Article sections required by the brief:** (a) Abstract, (b) Introduction, (c) Literature Survey, (d) Implementation, (e) Architecture diagram, (f) Algorithm, (g) Results and Discussion, (h) Conclusion, (i) References (IEEE or APA, consistent with Phase 1).
- **Presentation:** 8–12 December (the brief says 2025; probably a typo for 2026; confirm with the instructor).

## 2. Phase 1 (already submitted)

- Literature review of **15 papers (2021–2025)** on house price prediction, plus problem definition. The paper list and summaries are in the team's Phase 1 report (not included here: upload or paste it separately).
- One Phase 1 paper directly comparable with our results: **Sharma et al. (2024)** also used the Ames dataset and found **XGBoost** performed best.
- Phase 1 contribution claimed: a **fair comparison** of models, i.e. every model gets a comparable tuning budget.

**Three problem statements from Phase 1:**
1. Build an accurate **and** interpretable gradient-boosting model, explained with **SHAP**.
2. Measure how much **low-cost location features** (neighbourhood encoding, neighbourhood price lag) improve prediction.
3. Test transfer to the **Dubai** market: **out of scope for this submission**, listed as future work.

## 3. Task, data and environment

- **Task:** supervised **regression**: predict a house's sale price from its characteristics.
- **Dataset:** Ames Housing (Ames, Iowa, USA), Kaggle competition "House Prices: Advanced Regression Techniques"; compiled by De Cock (2011). Only Kaggle's `train.csv` is used (its `test.csv` has no prices): **1,460 houses × 81 columns** (79 features + `Id` + `SalePrice`).
- **Target:** `log(1 + SalePrice)`. SalePrice is right-skewed (skewness **1.88**); after the log transform the skewness is **0.12**. Predictions are converted back to dollars with `exp(x) − 1`.
- **Metrics:** RMSE on log price (also the Kaggle metric; ≈ relative error), MAE in USD, R², MAPE. Accuracy / precision / recall / F1 are **classification** metrics and do not apply to regression; the brief's "other relevant evaluation metrics" are used instead.
- **Environment:** Google Colab, CPU runtime, Python 3.13. Libraries: pandas 2.2.3, numpy 2.4.6, scikit-learn 1.6.1, xgboost 3.4.1, shap 0.54.1, matplotlib. `random_state = 42` for every random step.

## 4. Method (what the code does, in order)

1. **Load:** drop `Id` (row number); treat `MSSubClass` (dwelling-type code) as categorical.
2. **Split first (prevents data leakage):** stratified 80/20 train/test split on 10 quantile bins of log price. Train **1,168** houses, test **292**. Price distributions of the two sets are almost identical (mean $180,977 vs $180,698; median $163,000 vs $163,495).
3. **Outliers (training set only):** houses with above-ground living area > 4,000 sq ft **and** price < $300,000. Two training houses were flagged (Kaggle Ids 524 and 1299: 4,676 and 5,642 sq ft, sold for $184,750 and $160,000), both **partial sales** (sold before construction was finished), as described by De Cock (2011). Two other large houses ($755,000 and $745,000) follow the size–price trend and were kept. Training set after removal: **1,166** houses. The test set is not filtered.
4. **Preprocessing pipeline** (scikit-learn `Pipeline`, refitted inside every CV fold, so no statistic is learned from the houses being scored):
   - **Imputation** (custom transformer): "NA means absent" columns (pool, alley, fence, fireplace, garage and basement descriptors, masonry veneer type) → `"None"`; `MasVnrArea` → 0; `LotFrontage` (259 missing) → median of the same neighbourhood in the training data (fallback: overall median 70 ft); `Electrical` (1 missing) → mode (`SBrkr`); the one house with a basement but missing `BsmtExposure`/`BsmtFinType2` → mode (`No` / `Unf`); `GarageYrBlt` dropped (81 missing, near-duplicate of `YearBuilt`). 6,267 missing values filled in the training set; 0 remain in train or test.
   - **Ordinal encoding** of 10 quality/condition columns: None/Po/Fa/TA/Gd/Ex → 0–5.
   - **Feature engineering** (5 features): `TotalSF` = basement + 1st + 2nd floor area; `TotalBath` = full + 0.5·half baths (incl. basement); `HouseAge` = year sold − year built; `YearsSinceRemod` = year sold − year remodelled; `QualxSize` = overall quality × above-ground living area.
     - Correlation with log price (training set) vs the strongest ingredient: TotalSF 0.821 vs 0.641 (TotalBsmtSF); TotalBath 0.687 vs 0.582 (FullBath); QualxSize 0.834 vs 0.819 (OverallQual); HouseAge −0.581 vs 0.581 (YearBuilt, mirror image); YearsSinceRemod −0.578 vs 0.576 (YearRemodAdd, mirror image).
   - **One-hot encoding** of the remaining categorical columns (categories unseen in training → all zeros) and **standard scaling** of numeric columns.
   - Result: **277** model input columns (49 scaled numeric including the 5 engineered + 228 one-hot).
5. **Models, simple → complex,** each evaluated with the **same 5-fold cross-validation** on the training set (this is the validation step), with a common tuning budget of **30 candidate settings per tuned model**: Linear Regression (baseline, no tuning), Ridge (grid of 30 alphas, 0.01–1,000), Lasso (grid of 30 alphas, 0.0001–1), Random Forest (randomized search, 30 of 300 combinations), XGBoost (randomized search, 30 of 2,400 combinations).
   - Random Forest and XGBoost were each searched twice: the second RF search added `max_features` (not in the original plan of three parameters); the second XGBoost search widened two ranges whose best values had been at the edge. Both runs are reported.
6. **Model selection:** lowest mean CV RMSE → **Lasso**; refitted on all 1,166 training houses and evaluated **once** on the 292 test houses.
7. **Bias–variance analysis:** train vs CV RMSE for every model; learning curve for Lasso; validation curve for Ridge alpha.
8. **Location ablation (problem statement 2):** Lasso with Neighborhood (a) removed, (b) one-hot encoded, (c) target-encoded (Tier 1, scikit-learn `TargetEncoder`, cross-fitted to avoid leakage). Tier 2 (neighbourhood price lag) **not implemented**.
9. **Interpretation:** SHAP `TreeExplainer` on the tuned XGBoost model (summary/beeswarm plot + one waterfall plot).

## 5. Results

### 5a. Cross-validation (5-fold, training set, RMSE of log price; lower is better)

| Model | Best hyperparameters | Train RMSE | CV RMSE (mean ± std) | Gap (CV − train) |
|---|---|---|---|---|
| **Lasso (selected)** | alpha ≈ 0.000672 | 0.0992 | **0.1123 ± 0.0122** | 0.0131 |
| Ridge | alpha ≈ 18.87 | 0.0978 | 0.1148 ± 0.0131 | 0.0170 |
| XGBoost (final, widened ranges) | 1500 trees, learning rate 0.03, depth 3, subsample 0.8, colsample_bytree 0.1 | 0.0408 | 0.1156 ± 0.0093 | 0.0748 |
| Random Forest (final, incl. max_features) | 300 trees, min_samples_leaf 1, max_features 0.33, max_depth 30 | 0.0487 | 0.1312 ± 0.0131 | 0.0824 |
| Linear Regression | none | 0.0858 | 0.1324 ± 0.0100 | 0.0465 |
| *XGBoost, first run* | *1000 trees, rate 0.05, depth 3, subsample 0.7, colsample 0.3* | *0.0313* | *0.1161 ± 0.0106* | *0.0848* |
| *Random Forest, first run* | *100 trees, min leaf 2, max depth 15, all features per split* | *0.0602* | *0.1330 ± 0.0124* | *0.0728* |

Per-fold CV RMSE (same 5 folds for all models):
- Lasso 0.1252, 0.0947, 0.1119, 0.1036, 0.1261
- Ridge 0.1255, 0.0952, 0.1147, 0.1065, 0.1319
- XGBoost 0.1281, 0.1029, 0.1122, 0.1106, 0.1241
- Random Forest 0.1330, 0.1103, 0.1374, 0.1254, 0.1497
- Linear Regression 0.1415, 0.1383, 0.1335, 0.1131, 0.1354

Key observations:
- **Ranking:** Lasso < Ridge < XGBoost < Random Forest < Linear Regression. The top three differ by less than one fold standard deviation (~0.01), so they are close; Lasso beats Ridge on all 5 folds and XGBoost on 4 of 5.
- **Regularisation works:** Ridge and Lasso have higher training error than Linear Regression but lower CV error, better on all 5 folds (bias–variance trade-off). The same linear model family goes from last to first once a penalty is added.
- **Lasso feature selection:** 92 of 277 features kept (185 coefficients exactly 0). All 5 engineered features kept. Largest coefficients (log price per 1 std, or for the category): MSZoning_C (all) −0.240, Neighborhood_Crawfor +0.123, TotalSF +0.073, Neighborhood_StoneBr +0.068, Neighborhood_BrkSide +0.063, OverallQual +0.058, GrLivArea +0.053, Functional_Typ +0.052, HouseAge −0.049, MSZoning_RM −0.048. (One-hot and scaled numeric coefficients are on different scales; the number of houses in rare categories was not checked.)
- **Trees:** XGBoost (boosting) beat Random Forest (bagging) on all 5 folds. Both fit the training data far more closely than they generalise (largest gaps). Tuning `max_features` improved RF only slightly (0.1330 → 0.1312); widening XGBoost's ranges improved it only slightly (0.1161 → 0.1156).

### 5b. Final test-set evaluation (Lasso, 292 test houses, used once)

| Metric | Value |
|---|---|
| RMSE (log price) | **0.1096** (CV estimate 0.1123) |
| MAE | **$14,235** |
| R² (log price / USD) | 0.9284 / 0.9307 |
| MAPE | **8.22%** |

- Test RMSE is close to (slightly better than) the CV estimate → the CV estimate was reliable; model selection did not overfit the folds.
- Largest errors: four very cheap houses over-predicted by about 43–54% (actual $35,311 / $52,000 / $39,300 / $66,500 vs predicted $54,191 / $79,479 / $57,900 / $95,299); one $159,434 house predicted at $112,967. Two of the three most expensive test houses (about $500k and $610k) are predicted well below their price (about $440k and $475k, read from the plot); the most expensive (about $625k) almost exactly.
- Residuals are centred on 0, with a wider spread at low predicted prices.

### 5c. Bias–variance

- Gaps (CV − train): Lasso 0.013, Ridge 0.017, Linear Regression 0.047, XGBoost 0.075, Random Forest 0.082 → regularised linear models have low variance; tree ensembles have high variance (very low training error).
- **Learning curve (Lasso)**, training houses → train / CV RMSE: 186 → 0.0592 / 0.2623; 372 → 0.0833 / 0.1250; 559 → 0.0919 / 0.1176; 745 → 0.0983 / 0.1133; 932 → 0.0992 / 0.1123. The curves nearly converge; CV error still falls slightly at full size → more data would help a little; the remaining error is mostly bias.
- **Ridge validation curve:** CV error is U-shaped with its minimum at alpha ≈ 18.9. Small alpha → overfitting (low train, higher CV error); large alpha (1,000) → underfitting (both errors high).

### 5d. Location ablation (problem statement 2; Lasso with the same tuned alpha, 5-fold CV)

| Version | CV RMSE (mean ± std) | Change vs (a) | Per fold |
|---|---|---|---|
| (a) no location (Neighborhood dropped) | 0.1155 ± 0.0123 | — | 0.1284, 0.0980, 0.1147, 0.1066, 0.1296 |
| (b) Neighborhood one-hot | 0.1123 ± 0.0122 | −0.0032 (better on all 5 folds) | 0.1252, 0.0947, 0.1119, 0.1036, 0.1261 |
| (c) Neighborhood target encoding (Tier 1) | 0.1137 ± 0.0129 | −0.0017 (better on 4 of 5 folds) | 0.1270, 0.0940, 0.1106, 0.1084, 0.1286 |

- Version (b) reproduces Lasso's CV result exactly (built-in consistency check).
- Location gives a consistent but small improvement (~3% relative), smaller than the fold spread. A likely reason (not tested): quality, size, age, zoning and lot features already carry much of the location signal; Neighborhood is also used to impute LotFrontage in all three versions. Target encoding (1 column) recovers about half of the one-hot gain. Tier 2 price lag not implemented.

### 5e. SHAP (XGBoost, test set)

- Top features by mean |SHAP| (log price units): QualxSize 0.044, OverallQual 0.043, TotalSF 0.029, GrLivArea 0.028, BsmtQual 0.023, TotalBath 0.022, OverallCond 0.019, KitchenQual 0.014, Fireplaces 0.014, YearBuilt 0.014, ExterQual 0.014, LotArea 0.013, YearRemodAdd 0.012, GarageArea 0.012, 1stFlrSF 0.012.
- 3 of the top 6 are engineered features (QualxSize 1st, TotalSF 3rd, TotalBath 6th).
- Directions: high quality, size, bathrooms, fireplaces and build year → higher predicted price.
- No Neighborhood column in XGBoost's top 15 (consistent with the small ablation effect).
- Waterfall for one typical test house (Kaggle Id 630): actual $168,500, predicted $163,559; prediction 12.005 vs model average 12.026 (log scale); small positive (good basement exposure, wood-shingle roof, attached garage) and negative (basement and kitchen quality, overall quality, fireplaces) contributions nearly cancel. Feature values shown in the plot are standardised.
- SHAP shows what the model learned (association), not causation.

## 6. Points for the Introduction / motivation (facts only)

- House prices matter to buyers, sellers, lenders and planners; automated valuation needs accuracy **and** explanations people can trust (hence SHAP).
- Location is known to drive price, but rich location data (coordinates, amenities) is costly; problem statement 2 tests how much cheap location features (neighbourhood identity) add.
- The Ames dataset is a standard benchmark with 79 descriptive features, which makes results comparable with published work (e.g. Sharma et al., 2024).

## 7. Comparison with Phase 1 literature

- Sharma et al. (2024) found XGBoost best on Ames. In our setup (log target, engineered features, outlier removal, equal tuning budgets, identical CV folds), regularised **Lasso was slightly better than XGBoost**, though the top three models are within one fold standard deviation. Differences in preprocessing, tuning and evaluation could explain this; it was not investigated further.
- Other comparisons must come from the Phase 1 report (paper list not included here).

## 8. Limitations

- Single, small dataset (1,460 houses) from one US city, sold 2006–2010 (per De Cock, 2011; not re-checked in our code); no Dubai data (problem statement 3 is future work).
- No geographic coordinates, so no true spatial-lag features; Tier 2 neighbourhood price lag not implemented.
- Hyperparameter search limited to 30 candidates per model on CPU; XGBoost's best `colsample_bytree` (0.1) was at the edge of its range.
- Random Forest and XGBoost were re-tuned once after their first CV results (test set still unused; both runs reported).
- The ablation used Lasso's hyperparameters tuned with one-hot location for all three versions (not re-tuned per version).
- Ordinal quality encoding assumes equal steps between grades (affects linear models only).
- Very cheap and very expensive houses are predicted least accurately.

## 9. Future work

- Transfer to Dubai (problem statement 3), with a comparable Dubai dataset.
- Tier 2 neighbourhood price lag; true spatial features if coordinates become available.
- Wider tuning (e.g. XGBoost `colsample_bytree` below 0.1), stacking of Ridge + RF + XGBoost.

## 10. Files

**Figures** (`outputs/figures/`, 150 dpi):
| File | Content |
|---|---|
| `price_hist.png`, `log_price_hist.png` | SalePrice before / after log transform (EDA) |
| `train_grlivarea_vs_saleprice.png` | living area vs price, training set, outliers in red |
| `model_comparison_cv_rmse.png` | train vs CV RMSE, all models |
| `learning_curve_lasso.png` | learning curve (Lasso) |
| `validation_curve_ridge_alpha.png` | Ridge validation curve |
| `predicted_vs_actual_test.png` | predicted vs actual price, test set |
| `residuals_test.png` | residuals vs predicted price, test set |
| `ablation_location_cv_rmse.png` | location ablation |
| `shap_summary_xgboost.png` | SHAP beeswarm summary |
| `shap_waterfall_xgboost.png` | SHAP waterfall, one house |
| `architecture.png` | pipeline architecture diagram |

**Tables** (`outputs/results/`): `cv_results.csv`, `test_metrics.csv`, `ablation.csv`.
**Pseudocode:** `outputs/algorithm.md`. **Code:** `notebooks/02_modelling.ipynb` (EDA: `notebooks/01_eda.ipynb`).
