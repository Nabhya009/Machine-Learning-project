# Algorithm: house-price prediction pipeline

Stepwise pseudocode of `notebooks/02_modelling.ipynb`, in the order the code runs.
Notation: D = data set, y = log(1 + SalePrice), K = 5 folds, B = 30 candidate settings per tuned model, seed = 42.

```
INPUT:  Ames Housing train.csv (1,460 houses, 79 features + SalePrice)
OUTPUT: selected model M*, its test metrics, CV results table, location ablation, SHAP explanations

 1  LOAD
 2    D ← read train.csv; drop Id; cast MSSubClass to text (categorical)
 3    X ← D without SalePrice;  y ← log(1 + SalePrice)

 4  SPLIT (before any statistic is learned, to prevent leakage)
 5    bins ← 10 quantile bins of y
 6    (X_train, y_train), (X_test, y_test) ← stratified split of (X, y) on bins, 80 % / 20 %, seed
 7    lock (X_test, y_test) until step 33

 8  REMOVE OUTLIERS (training set only)
 9    drop training houses with GrLivArea > 4000 sq ft AND SalePrice < $300,000

10  DEFINE PREPROCESSING P(location)          ▷ a scikit-learn Pipeline; nothing is fitted yet
11    impute:   fit → learn per-Neighborhood median LotFrontage, overall median, modes (from data passed to fit)
12              transform → drop GarageYrBlt; basement-mode fix; "NA = absent" columns → "None";
13                          MasVnrArea → 0; LotFrontage → neighbourhood median (fallback: overall median);
14                          Electrical → mode
15    quality:  map None/Po/Fa/TA/Gd/Ex → 0..5 in the 10 quality columns
16    features: add TotalSF, TotalBath, HouseAge, YearsSinceRemod, QualxSize
17    location: "onehot" → Neighborhood treated like other text columns
18              "none"   → drop Neighborhood
19              "target" → Neighborhood → TargetEncoder (cross-fitted) → standard scaling
20    encode:   numeric columns → StandardScaler (mean / std learned in fit);
21              text columns → one-hot (categories learned in fit; unseen → all zeros)

22  CROSS-VALIDATION SET-UP
23    folds ← KFold(K = 5, shuffle, seed)  ▷ the same folds for every model

24  TRAIN AND TUNE EACH MODEL m ∈ {Linear, Ridge, Lasso, Random Forest, XGBoost}
25    for each candidate setting θ of m (Linear: 1 setting; others: B = 30, grid or random sample, seed):
26      for k = 1..K:
27        fit P(onehot) + m(θ) on the K − 1 training folds        ▷ preprocessing refitted per fold
28        RMSE_train[k] ← error on those training folds
29        RMSE_cv[k]    ← error on held-out fold k
30      score(θ) ← mean_k RMSE_cv[k]
31    θ*_m ← argmin_θ score(θ);  refit P(onehot) + m(θ*_m) on all of X_train
32    record train RMSE, CV RMSE (mean ± std), gap = CV − train, θ*_m

33  SELECT AND TEST (the only use of the test set)
34    M* ← model m with the lowest mean CV RMSE
35    ŷ ← M*.predict(X_test)
36    report RMSE(y_test, ŷ) on log scale;
37           MAE, R², MAPE on dollars after exp(·) − 1  (and R² on log scale)

38  BIAS–VARIANCE ANALYSIS (training set only)
39    learning curve of M*: train / CV RMSE for 20 %, 40 %, …, 100 % of each fold's training part
40    validation curve of Ridge: train / CV RMSE for all 30 alphas (from step 25 scores)

41  LOCATION ABLATION (training set only)
42    for location ∈ {none, onehot, target}:
43      5-fold CV of P(location) + M* with M*'s tuned hyperparameters
44    report CV RMSE of each version and its change relative to "none"

45  INTERPRETATION
46    explainer ← SHAP TreeExplainer(tuned XGBoost)
47    SHAP values for the preprocessed test houses → summary (beeswarm) plot and
48    waterfall plot for the test house whose prediction is closest to the median prediction
```

**Key properties:**
- **No leakage:** every learned statistic (medians, modes, category lists, scaling, target encodings) is fitted only on the data passed to `fit`, i.e. on the training folds inside CV, and on the full training set for the final model. The test set is used once, in steps 35–37.
- **Fair comparison:** all models use the same preprocessing, the same 5 folds, and up to 30 candidate settings each.
- **Re-tuning note:** Random Forest and XGBoost were each searched twice. The second search added `max_features` (RF) or widened two ranges (XGBoost) after the first search's results. Both runs are recorded in the notebook; the second search's result is the one compared.
