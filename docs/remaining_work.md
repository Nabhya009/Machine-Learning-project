# What's left, and how to do it

Status at 17:30, 20 Nov 2026. **Deadline: midnight tonight.** About 6.5 hours left.

The **code is finished**: notebook run, all figures and tables saved, README, requirements and pseudocode written. What remains is the **article**, the **slides**, the **code ZIP** and the **upload**, all of which are the team's work.

Facts and numbers for everything below are in `docs/project_context.md` (§ = section of that file).

---

## 0. Suggested timeline

| Time | Task | Who (suggestion) |
|---|---|---|
| 17:30–17:45 | Commit the code; make and check the code ZIP (§3 below) | Nabhya |
| 17:30–21:30 | Article, sections split between members (§1) | everyone in parallel |
| 19:30–21:30 | Slides (§2), built from the article's figures | one member |
| 21:30–22:30 | Merge the article, format it, check the page count, references | Nabhya |
| 22:30–23:15 | Proofread; check every number against `project_context.md` | two members |
| 23:15–23:40 | Export the PDF; upload all three files | Nabhya |
| 23:40 | Buffer for upload problems | — |

Split the article by section so people work in parallel. One person (the lead) merges and formats.

---

## 1. Article (10–15 pages, Times New Roman 12 pt, 1.5 line spacing)

### Rules first
- **Write everything in your own words.** Turnitin checks for plagiarism and AI writing. Do **not** paste text from the notebook's markdown cells, the README, `project_context.md` or `algorithm.md`. Those were written with AI help, and Turnitin may flag them. Use them for facts only.
- **Rephrase the Phase 1 material.** Turnitin compares against earlier submissions, including your own Phase 1 report.
- **Every number must match** `outputs/results/*.csv` or `project_context.md`. Round consistently (4 decimals for RMSE, 2 for R², whole dollars for MAE).
- Number figures and tables ("Fig. 3", "Table 2"), refer to each in the text, and give each a caption.

### Section-by-section plan

| Section | Target length | What to include | Sources |
|---|---|---|---|
| **(a) Abstract** | ~200 words | Problem; data (Ames, 1,460 houses); method (5 models, same 5-fold CV, 30-candidate budget, leakage-free pipeline); main result (Lasso: test RMSE 0.1096, MAE $14,235, R² 0.93, MAPE 8.2%); location adds ~3%; SHAP top features. **Write it last.** | §5 |
| **(b) Introduction** | 1 page | Why price prediction matters; why accuracy *and* interpretability; why cheap location features; the 3 problem statements (Dubai = future work); the paper's contributions; outline of the paper | §2, §6 |
| **(c) Literature Survey** | 2–2.5 pages | Condensed from Phase 1 (15 papers): group by theme (linear vs tree ensembles; interpretability/SHAP; location features), not paper-by-paper; end with the gap you address | Phase 1 report |
| **(d) Implementation** | 1.5–2 pages | Environment and library versions; data split (stratified, 1,168/292); outlier rule and why; each preprocessing step and **why**; the 5 engineered features; CV set-up; models and tuning grids (a table works well); the metrics and **why not accuracy/F1** | §3, §4 |
| **(e) Architecture diagram** | ~1 page | `architecture.png` with a caption, plus 1 paragraph walking through it | Fig: `architecture.png` |
| **(f) Algorithm** | ~1 page | Pseudocode **rewritten in your own format** from `outputs/algorithm.md`, plus a short explanation of the train / validation (CV) / test logic and why there's no leakage | `algorithm.md` |
| **(g) Results and Discussion** | 3–4 pages | See the structure below | §5, §7 |
| **(h) Conclusion** | 0.5–1 page | Answer each problem statement in 1–2 sentences; limitations; future work (Dubai, Tier 2, wider tuning) | §8, §9 |
| **(i) References** | ~1 page | Same style as Phase 1 (IEEE or APA). Must include: De Cock (2011); the Kaggle dataset; Sharma et al. (2024); scikit-learn; XGBoost (Chen & Guestrin, 2016); SHAP (Lundberg & Lee, 2017); your Phase 1 papers that you cite | — |

### Suggested structure for (g) Results and Discussion
1. **Model comparison:** Table (from `cv_results.csv`) + `model_comparison_cv_rmse.png`. The ranking, and how close the top three are (within one fold std).
2. **Bias–variance and regularisation:** the train-vs-CV gaps; `validation_curve_ridge_alpha.png` (U-shape); `learning_curve_lasso.png` (curves converge, mostly bias left). Also how regularisation took the linear model from last to first.
3. **Final test result:** table from `test_metrics.csv`; test RMSE close to CV, so the estimate was reliable; `predicted_vs_actual_test.png` and `residuals_test.png`; where the model fails (very cheap and very expensive houses).
4. **Feature selection and engineered features:** Lasso keeps 92 of 277; all 5 engineered features kept; correlations (§4).
5. **Location ablation (problem statement 2):** table + `ablation_location_cv_rmse.png`; small but consistent gain; why it's small; one-hot vs target encoding.
6. **Interpretation (problem statement 1):** `shap_summary_xgboost.png` + `shap_waterfall_xgboost.png`; top features; engineered features rank high; association, not causation.
7. **Comparison with literature:** Sharma et al. (2024) found XGBoost best, while we found Lasso slightly ahead. Discuss possible reasons (log target, feature engineering, regularisation, small differences within the noise). Add comparisons with other Phase 1 papers if they report Ames results.

### Page-count check
The figures take space. If you go over 15 pages, put the two SHAP plots side by side, or the learning and validation curves side by side. If you're under 10, expand the discussion, not the method.

---

## 2. Slides (`YourID_Phase2_Slides.pptx`)

The brief asks for: the problem statement, architecture + algorithm, results + discussion, conclusion. Suggested 12 slides:

| # | Slide | Content / figure |
|---|---|---|
| 1 | Title | Project title, team names and IDs, course |
| 2 | Problem and motivation | Why price prediction; accuracy + interpretability; the 3 problem statements |
| 3 | Data | Ames, 1,460 houses, 79 features; `log_price_hist.png` (skew 1.88 → 0.12) |
| 4 | Preprocessing | Split first (no leakage); `train_grlivarea_vs_saleprice.png` (2 partial-sale outliers); imputation rules, encoding, 5 engineered features |
| 5 | Architecture | `architecture.png` |
| 6 | Algorithm / training protocol | Same 5 folds, 30 candidates per model, select by CV, test once (short pseudocode) |
| 7 | Model comparison | `model_comparison_cv_rmse.png` + the 5 CV RMSE values |
| 8 | Bias–variance | `validation_curve_ridge_alpha.png` + `learning_curve_lasso.png` |
| 9 | Test results | Metrics table + `predicted_vs_actual_test.png` |
| 10 | Location ablation | `ablation_location_cv_rmse.png`; ~3% gain |
| 11 | Interpretation | `shap_summary_xgboost.png` (optionally the waterfall) |
| 12 | Conclusion | Answers to the problem statements, limitations, future work (Dubai) |

Keep the text short: numbers and figures on the slides, explanations spoken.

---

## 3. Code ZIP (`YourID_Code.zip`)

1. **Commit first.** Suggested message:
   `feat: add XGBoost, model comparison, test evaluation, ablation, SHAP; add outputs, README and docs`
2. From the project folder, in PowerShell (replace `YourID`):
   ```powershell
   Compress-Archive -Path README.md, requirements.txt, notebooks, outputs -DestinationPath YourID_Code.zip
   ```
3. **Open the ZIP and check it** before uploading:
   - ✅ `README.md`, `requirements.txt`, `notebooks/01_eda.ipynb`, `notebooks/02_modelling.ipynb`, `outputs/figures/*.png`, `outputs/results/*.csv`, `outputs/algorithm.md`
   - ❌ no `data/` or `train.csv` (Kaggle rules), no `.git`, `.venv`, `.claude`, `__pycache__`
   - `docs/` is left out on purpose: it holds working notes for the article, not code.

---

## 4. Final submission checklist

- [ ] Article: all 9 sections (a–i), 10–15 pages, Times New Roman 12, 1.5 spacing, figures captioned and referenced
- [ ] Article states that accuracy/precision/recall/F1 don't apply (regression) and which metrics are used instead
- [ ] Article covers preprocessing, the train/validation(CV)/test split, bias–variance, regularisation and model selection (the brief's requirements)
- [ ] Every number checked against the CSVs / `project_context.md`
- [ ] References complete, one consistent style
- [ ] Article in your own words (Turnitin)
- [ ] File names: `YourID_Article.pdf`, `YourID_Code.zip`, `YourID_Phase2_Slides.pptx`
- [ ] Code ZIP contents checked (§3)
- [ ] All three files uploaded to LMS **before midnight**, and the upload confirmed

---

## 5. After the deadline: presentation prep (8–12 Dec)

Everyone should be able to answer these:
1. Why split before preprocessing? What is data leakage?
2. Why log price? What do RMSE 0.11 on log price and MAPE 8.2% mean in plain words?
3. Why not accuracy or F1?
4. Why remove only 2 outliers, and only from the training set?
5. Why median-by-neighbourhood for `LotFrontage`? Why `"None"` for pools and garages?
6. Ridge vs Lasso: what is the difference, and why did Lasso drop 185 features?
7. Bagging vs boosting: why did XGBoost beat Random Forest?
8. Why did Lasso beat XGBoost, when Sharma et al. found the opposite?
9. What does the learning curve tell you? What does the validation curve tell you?
10. How does `TargetEncoder` avoid leakage? Why is the location effect small?
11. What is a SHAP value? Why is `QualxSize` top? Does SHAP show causation?
12. Why were RF and XGBoost re-tuned, and why is that not cheating?
13. Limitations, and how you would extend the work to Dubai.

The notebook's markdown cells explain each of these. Read them before the presentation.
