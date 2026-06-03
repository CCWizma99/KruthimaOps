# ML Hackathon Validation Report

This report summarizes the performance metrics, feature importances, and validation check outcomes for the Sri Lanka flood risk prediction pipeline.

---

## 1. Fold-by-Fold Performance Metrics

The fold-by-fold Out-Of-Fold (OOF) cross-validation scores for the three CatBoost models are detailed below.

### CatBoost Full v2
* **Average MAE:** `0.179101`
* **Average RMSE:** `0.234686`

| Fold | MAE | RMSE |
| :--- | :---: | :---: |
| Fold 1 | 0.179747 | 0.235283 |
| Fold 2 | 0.179267 | 0.235766 |
| Fold 3 | 0.179864 | 0.236640 |
| Fold 4 | 0.178171 | 0.232969 |
| Fold 5 | 0.178453 | 0.232773 |

### CatBoost Full v3
* **Average MAE:** `0.179129`
* **Average RMSE:** `0.234691`

| Fold | MAE | RMSE |
| :--- | :---: | :---: |
| Fold 1 | 0.179671 | 0.235169 |
| Fold 2 | 0.179581 | 0.235791 |
| Fold 3 | 0.179959 | 0.236583 |
| Fold 4 | 0.178055 | 0.232942 |
| Fold 5 | 0.178380 | 0.232772 |

### CatBoost Full RankGauss
* **Average MAE:** `0.179062`
* **Average RMSE:** `0.234953`

| Fold | MAE | RMSE |
| :--- | :---: | :---: |
| Fold 1 | 0.179902 | 0.235650 |
| Fold 2 | 0.179612 | 0.236203 |
| Fold 3 | 0.179674 | 0.236597 |
| Fold 4 | 0.177911 | 0.233085 |
| Fold 5 | 0.178212 | 0.233229 |

---

## 2. Top 10 Feature Importances (v3 Baseline)

The top 10 features driving predictions in the `outputs_catboost_full_v3` run are ranked by importance below:

| Rank | Feature | Importance | Description |
| :---: | :--- | :---: | :--- |
| 1 | `district_target_enc` | 21.84% | Bayesian-smoothed mean risk score by district |
| 2 | `reason_not_good_to_live` | 6.42% | Survey indicator for uninhabitable location reasons |
| 3 | `grid_id_target_enc` | 4.96% | Bayesian-smoothed mean risk score by 0.5-degree grid |
| 4 | `distance_to_river_m_log1p` | 4.73% | Log-transformed river proximity |
| 5 | `distance_to_river_m` | 4.35% | Distance to nearest river in meters |
| 6 | `inundation_area_sqm` | 3.47% | Survey indicator for flooded area in square meters |
| 7 | `inundation_ratio` | 3.16% | Flooded area ratio relative to landcover class averages |
| 8 | `inundation_area_log` | 3.13% | Log-transformed flooded area size |
| 9 | `infrastructure_score` | 3.10% | Regional infrastructure quality metric |
| 10 | `rainfall_7d_mm_log1p` | 3.09% | Log-transformed preceding 7-day rainfall |

---

## 3. Validation Check Outcome

| Check | Status | Verification Detail |
| :--- | :---: | :--- |
| File Paths | ✅ Pass | All required directories and scripts exist. |
| Required Packages | ✅ Pass | All Python dependencies are importable. |
| CSV Columns | ✅ Pass | Column headers match `[record_id, flood_risk_score]`. |
| Row Count | ✅ Pass | Length matches sample submission exactly (5,300 rows). |
| Missing Values | ✅ Pass | Zero NaN/null entries in prediction values. |
| Duplicate IDs | ✅ Pass | No duplicate `record_id` strings found. |
| Prediction Range | ✅ Pass | Predictions capped within `[0.348103, 0.637862]`. |
| Order Alignment | ✅ Pass | Matches `sample_submission.csv` record-by-row sequence. |
| Final submission | ✅ Pass | Identical to `submission_blend_best_mae.csv`. |
