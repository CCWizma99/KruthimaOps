# ML Hackathon Validation Report (Upgraded Pipeline)

This report summarizes the performance metrics, feature importances, and validation check outcomes for the upgraded Sri Lanka flood risk prediction pipeline, including the new model diversity experiments and diverse blend.

---

## 1. Fold-by-Fold Performance Metrics

The fold-by-fold Out-Of-Fold (OOF) cross-validation scores for the upgraded models (1 seed, 5 folds) are detailed below.

### CatBoost Full Clean (No Pseudo-labeling)
* **Average MAE:** `0.17882`
* **Average RMSE:** `0.23465`
* **Average EV:** `0.03472`
* **Average Score:** `0.38399`

| Fold | MAE | RMSE | EV |
| :--- | :---: | :---: | :---: |
| Fold 0 | 0.17943 | 0.23545 | 0.03407 |
| Fold 1 | 0.17927 | 0.23571 | 0.03726 |
| Fold 2 | 0.17966 | 0.23666 | 0.03441 |
| Fold 3 | 0.17740 | 0.23261 | 0.03351 |
| Fold 4 | 0.17830 | 0.23276 | 0.03431 |

### CatBoost Safe Pseudo-labeled
* **Average MAE:** `0.17869`
* **Average RMSE:** `0.23452`
* **Average EV:** `0.03580`
* **Average Score:** `0.38374`

| Fold | MAE | RMSE | EV |
| :--- | :---: | :---: | :---: |
| Fold 0 | 0.17958 | 0.23543 | 0.03338 |
| Fold 1 | 0.17901 | 0.23559 | 0.03759 |
| Fold 2 | 0.17976 | 0.23655 | 0.03529 |
| Fold 3 | 0.17721 | 0.23256 | 0.03398 |
| Fold 4 | 0.17788 | 0.23241 | 0.03734 |

### CatBoost Full Pseudo-labeled
* **Average MAE:** `0.17832`
* **Average RMSE:** `0.23423`
* **Average EV:** `0.03814`
* **Average Score:** `0.38314`

| Fold | MAE | RMSE | EV |
| :--- | :---: | :---: | :---: |
| Fold 0 | 0.17901 | 0.23503 | 0.03672 |
| Fold 1 | 0.17849 | 0.23503 | 0.04196 |
| Fold 2 | 0.17903 | 0.23609 | 0.03911 |
| Fold 3 | 0.17738 | 0.23266 | 0.03316 |
| Fold 4 | 0.17768 | 0.23232 | 0.03815 |

### LightGBM Full Clean
* **Average MAE:** `0.179193`
* **Average RMSE:** `0.234913`
* **Average EV:** `0.032109`
* **Average Score:** `0.384565`

| Fold | MAE | RMSE | EV |
| :--- | :---: | :---: | :---: |
| Fold 0 | 0.17983 | 0.23560 | 0.03144 |
| Fold 1 | 0.17876 | 0.23524 | 0.03923 |
| Fold 2 | 0.18036 | 0.23707 | 0.03064 |
| Fold 3 | 0.17844 | 0.23357 | 0.02639 |
| Fold 4 | 0.17861 | 0.23306 | 0.03362 |

### XGBoost Full Clean
* **Average MAE:** `0.181111`
* **Average RMSE:** `0.236308`
* **Average EV:** `0.020529`
* **Average Score:** `0.387566`

| Fold | MAE | RMSE | EV |
| :--- | :---: | :---: | :---: |
| Fold 0 | 0.18156 | 0.23675 | 0.02169 |
| Fold 1 | 0.18117 | 0.23725 | 0.02274 |
| Fold 2 | 0.18232 | 0.23846 | 0.01946 |
| Fold 3 | 0.17978 | 0.23455 | 0.01753 |
| Fold 4 | 0.18072 | 0.23447 | 0.02110 |

---

## 2. Final Blended Metrics

We performed a diverse blend optimizing the real competition score:

* **MAE:** `0.175220`
* **RMSE:** `0.231586`
* **Explained Variance (EV):** `0.033410`
* **Competition Score:** `0.378223` (Beats current best v2 score `0.381170`!)
* **Prediction Range:** `[0.3430, 0.6323]` (OOF predictions), `[0.3569, 0.6139]` (test predictions)

### Optimal Diverse Blending Weights:
* `catboost_full_clean`: **0.026645**
* `catboost_safe_pseudo`: **0.127408**
* `catboost_full_pseudo`: **0.689818**
* `lightgbm_full_clean`: **0.156130**
* `xgboost_full_clean`: **0.000000**

---

## 3. Validation Check Outcome

All validation checks on the final submission file `final_submission_v3.csv` have passed successfully:

| Check | Status | Verification Detail |
| :--- | :---: | :--- |
| File Paths | ✅ Pass | All upgraded modules exist and are runnable. |
| Required Packages | ✅ Pass | All Python dependencies import correctly. |
| CSV Columns | ✅ Pass | Column headers match `[record_id, flood_risk_score]`. |
| Row Count | ✅ Pass | Length matches sample submission exactly (5,300 rows). |
| Missing Values | ✅ Pass | Zero NaN/null entries in prediction values. |
| Duplicate IDs | ✅ Pass | No duplicate `record_id` strings found. |
| Prediction Range | ✅ Pass | Predictions lie within `[0.3569, 0.6139]`. |
| Order Alignment | ✅ Pass | Matches `sample_submission.csv` record-by-row sequence. |
| Final submission | ✅ Pass | Saved successfully as `final_submission_v3.csv`. |
