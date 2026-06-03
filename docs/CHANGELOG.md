# ML Opsidian: Genesis — Change Log

> All agent actions are recorded here in chronological order.
> train.csv and test.csv are NEVER modified. All derived files are listed here.

---

## [2026-05-30] — v1: Initial Baseline (Initial_trainer.py)

### What existed
- Single XGBoost model, 5-fold KFold
- No feature engineering beyond raw columns
- No early stopping wired correctly
- Global OOF metrics only (no per-fold reporting)
- Paths hardcoded to /kaggle/input/ — not locally runnable

### Status: Archived (not run locally)

---

## [2026-05-30] — v2: Full Ensemble Pipeline (train_v2.py)

### What we built
- 3-model ensemble: XGBoost + LightGBM + CatBoost
- 5-fold KFold (shuffle=True, seed=42)
- 10 new engineered interaction features
- Per-fold MAE, RMSE, Explained Variance reporting
- Category alignment: train+test union cast (prevents unseen cat errors)
- Boundary clip: predictions clipped to [0.0, 1.0]
- Outputs: submission_v2.csv + fold_report_v2.csv

### 10 Features Added
| Feature | Formula |
|---|---|
| `river_rain_interaction` | log1p(dist_river) x log1p(rain_7d) |
| `river_monthly_exposure` | log1p(dist_river) x log1p(monthly_rain) |
| `elev_rain_risk` | elev_yeo / (rain_7d_log1p + eps) |
| `water_signal` | ndwi_qmap.clip(lower=0) |
| `drainage_deficit` | (rain_log+1) x (1 - drainage_yeo) |
| `infra_resilience` | infra_score / pop_density_log |
| `evacuation_difficulty` | hospital_log + evac_log |
| `inundation_density_risk` | log1p(inundation_sqm) / pop_density_log |
| `terrain_veg_risk` | roughness x (1 - ndvi_qmap) |
| `flood_pressure` | extreme_weather_idx x seasonal_idx.clip(0) |

### Bugs Encountered & Fixed

#### Bug 1 — XGBoost 3.x API Change
- Error: `TypeError: XGBModel.fit() got an unexpected keyword argument 'early_stopping_rounds'`
- Root cause: XGBoost 3.x moved `early_stopping_rounds` from `.fit()` to the constructor
- Fix: Moved `early_stopping_rounds=50` into `XGBRegressor(...)` constructor

#### Bug 2 — CatBoost NaN in Categorical Columns
- Error: `CatBoostError: Invalid type for cat_feature NaN — must be integer or string`
- Root cause: Categorical columns had NaN values; CatBoost refuses NaN in cat columns
- Fix: Added `.fillna("missing").astype(str)` before category alignment for all cat columns
- Note: XGBoost and LightGBM handle NaN natively — only CatBoost needed this

#### Bug 3 — Emoji UnicodeEncodeError on cp1252 terminal
- Error: `UnicodeEncodeError: 'charmap' codec can't encode character '\u2705'`
- Root cause: Python 3.14 on Windows uses cp1252 terminal encoding — no emoji support
- Fix: Replaced all emoji (rocket, checkmark, chart) with ASCII tags [LOAD], [FEAT], [DONE]

### v2 Results (fold_report_v2.csv)
| Fold | MAE | RMSE | EV |
|---|---|---|---|
| 1 | 0.1792 | 0.2346 | 0.0357 |
| 2 | 0.1783 | 0.2346 | 0.0262 |
| 3 | 0.1808 | 0.2361 | 0.0369 |
| 4 | 0.1794 | 0.2349 | 0.0260 |
| 5 | 0.1795 | 0.2349 | 0.0268 |
| **OVERALL** | **0.1794** | **0.2350** | **0.0303** |

### v2 Diagnosis — What Went Wrong
1. **EV = 0.03 — near-zero.** Model is a de-facto mean predictor.
2. **Prediction range [0.38, 0.61]** — collapsed to mean (0.478). No spread.
3. **All features have r < 0.08 correlation** with target — signal is purely non-linear/interaction-driven.
4. **Early stopping fired too early** — XGB stopped at iter ~121 (lr=0.02 too slow).
5. **KFold not stratified** — bell-shaped target distribution not guaranteed across folds.
6. **Equal ensemble weights** — CatBoost found 427 iterations vs XGB 121; likely higher quality but weighted equally.
7. **No target encoding** — district used as raw label, losing geographic risk context.

### v2 Key Learnings
- Weak individual correlations do NOT mean the problem is hard — they mean trees need
  more room (higher lr, deeper interactions) to find the non-linear patterns.
- CatBoost consistently found more useful iterations (427 vs ~120) suggesting it handles
  this problem better. Should get higher ensemble weight in v3.
- The prediction spread problem needs BOTH better features AND calibration.

### v2 Submission: submission_v2.csv (5300 rows, range [0.38, 0.61])

---

## [2026-05-30] — v3: Phase 1 Improvements (train_v3.py)

### What changed from v2

| Area | v2 | v3 |
|---|---|---|
| CV Strategy | KFold random | StratifiedKFold on 10-bin target |
| Learning rate | 0.02 (too slow) | 0.05 (faster exploration) |
| Early stop patience | 50 | 100 (more patience) |
| Ensemble weights | Equal 1/3 each | Inverse-RMSE weighted |
| Target encoding | None | KFold-safe district + lat/lon grid encoding |
| New features | 10 | +6 more (16 total) |
| Calibration | None | Isotonic regression on OOF |

### 6 New Features Added in v3
| Feature | Formula | Rationale |
|---|---|---|
| `is_repeat_flood_zone` | historical_flood_count > 2 | Binary: chronic flood area |
| `rain_spike_ratio` | rain_7d / (monthly_rain + eps) | Sudden vs sustained rainfall |
| `confirmed_risk` | flood_occurred=Yes AND is_good_to_live=No | Both risk signals true |
| `vulnerability` | evac_difficulty x pop_density / (infra+1) | Isolated + dense + poor infra |
| `district_target_enc` | Mean flood_risk_score per district (CV-safe) | Geographic risk context |
| `grid_target_enc` | Mean flood_risk_score per 0.5deg lat/lon grid (CV-safe) | Spatial risk zone |

### Status: Testing High-Variance Architecture (v7)

#### The v7 "Variance Inflation" Architecture
- **Problem**: Models consistently detect the 97% noise ratio and intentionally default to predicting a narrow range `[0.38, 0.60]` (the mean) to minimize their RMSE penalty. The competition metric actively penalizes this flat-line behavior by scaling up the foundational error when prediction variance is low.
- **Solution**: Implemented architectures explicitly designed to prevent early convergence and artificially widen the prediction distribution (Phase 4 of playbook).
- **Components**:
  1. **XGBoost (DART)**: Uses `booster='dart'` (Dropouts meet Multiple Additive Regression Trees). By randomly dropping trees during boosting, the model is forced to learn multiple independent pathways, preventing it from converging to a single narrow mean.
  2. **ExtraTreesRegressor**: Replaces traditional gradient boosting with highly randomized splitting thresholds. Extremely effective on noisy datasets, naturally inflating the variance of the predictions.
  3. **Micro-Learning Rates**: LightGBM and CatBoost learning rates crashed from `0.05` to `0.005`/`0.01`, with estimators pushed to `5000` to capture tiny fragments of the remaining 3% signal without panicking and stopping early.

#### The v6 "Ground-Truth Isolation" Architecture
- **Problem**: EV artificially capped at ~3% because we were feeding the models 20,084 rows of synthetic noise alongside 802 rows of real data.
- **Solution**: Shifted back to the highly optimized tree ensemble (v3.5) but implemented strict sample weighting based on the `is_synthetic` backend flag.
- **Components**:
  1. **Sample Weighting**: Extracted the `is_synthetic` column *before* feature matrix construction. Assigned a massive `50.0` sample weight to real rows (`is_synthetic=NaN`) and `1.0` to synthetic noise rows (`is_synthetic=True`).
  2. **Tree Ensembles**: Passed `sample_weight` directly into the `.fit()` functions of XGBoost, LightGBM, and CatBoost. This mathematically forces the gradients to prioritize minimizing loss on the 802 real Sri Lankan locations, effectively ignoring the noise injection.

#### The v5 "Emergency Pivot" Architecture
- **Problem**: In v4, EV mathematically capped at ~3.2% because the target is 97% noise. Tree ensembles (XGB, LGB) overfit immediately on blocky splits and stop iterating to protect their RMSE.
- **Solution**: Shifted entirely away from trees to a continuous, differentiable architecture.
- **Components**:
  1. **Lasso (L1) Feature Pruning**: Fits a strong regularized linear model inside the fold to dynamically drop useless noisy features, forcing the model to focus only on true signal.
  2. **StandardScaler + One-Hot Encoding**: Required preprocessing for neural networks.
  3. **MLPRegressor (Deep Neural Network)**: 2 hidden layers (64, 32) using Adam optimizer and heavy L2 regularization, attempting to learn a smooth continuous equation instead of orthogonal tree splits.
  4. **Ridge Regression**: Kept as a simple linear baseline.

### Bugs Encountered & Fixed

#### Bug 1 — Pandas 3.x IntCastingNaNError on lat/lon grid binning
- Error: `pandas.errors.IntCastingNaNError: Cannot convert non-finite values (NA or inf) to integer`
- Line: `df["lat_bin"] = (df["latitude"] / 0.5).astype(int)`
- Root cause: `latitude` column has NaN values. Pandas 3.x refuses to cast float NaN to int.
- Fix: Added `.fillna(df["latitude"].median())` before the int cast.

#### Bug 2 — XGBoost 3.x Rejects Categorical Columns with Float Code Dtype
- Error: `XGBoostError: Category index from DataFrame has floating point dtype, consider using strings or integers instead`
- Root cause: XGBoost 3.x strictly requires that `pd.Categorical` columns have integer
  (int8/int16/int32) codes internally. When you use `pd.Categorical(series, categories=...)`
  constructor and then slice with `.iloc[idx].copy()`, pandas can lose the integer code
  guarantee and fall back to float codes. XGBoost 3.x catches this and raises an error.
- Fix (two-part):
  1. Changed categorical creation from `pd.Categorical(col, categories=...)` to
     `pd.CategoricalDtype(categories=all_vals) + .astype(cdt)` — this reliably produces
     integer codes and stores the dtype in `cat_dtype_map` for reuse.
  2. After every `.iloc` fold slice, re-cast all categorical columns via
     `.astype(str).astype(cdt)` to guarantee clean integer codes for XGBoost.
- Note: LightGBM and CatBoost are unaffected — they handle categorical differently internally.

#### Bug 3 — Pandas Categorical Dtype Inheritance on .map()
- Error: `ValueError: DataFrame.dtypes for data must be int, float, bool or category. Invalid columns: district_target_enc: category`
- Root cause: When computing target encoding, `tr_rows["district"].map(dist_enc)` was used. Because `district` was cast to `pd.CategoricalDtype` earlier, Pandas 3.x *inherits* the Categorical dtype for the new mapped column, making it a category instead of a float. The `to_xgb` converter then failed because it expected only explicitly named categorical features.
- Fix:
  1. Cast to string and explicitly to float during mapping: `tr_rows["district"].astype(str).map(dist_enc).fillna(GLOBAL_MEAN).astype(float)`.
  2. Upgraded `to_xgb_fmt()` to dynamically catch *any* column with `hasattr(col, "cat")` instead of relying on a hardcoded list, completely bulletproofing the XGBoost pipeline against rogue categorical columns.

### v3 Results
| Metric | Raw Ensemble | After Calibration |
|---|---|---|
| MAE | 0.1796 | 0.1793 |
| RMSE | 0.2351 | 0.2348 |
| EV | 0.0295 | 0.0320 |

**Diagnosis:** The new features are dominating the top of the importance chart (`district_target_enc`, `reason_not_good_to_live`, `confirmed_risk` are top 3). However, EV is *still* stuck at ~3%. The models are halting extremely early (XGB iter 30-60). This indicates the signal-to-noise ratio is so poor that any deviation from predicting the mean *increases* RMSE on the validation set. We need more aggressive feature combinations (Phase 2).


---

## File Inventory

| File | Status | Notes |
|---|---|---|
| `train.csv` | READ-ONLY | Never modified |
| `test.csv` | READ-ONLY | Never modified |
| `sample_submission.csv` | READ-ONLY | Column format reference |
| `AGENTS.md` | READ-ONLY | Agent directives and hard rules |
| `Initial_trainer.py` | Archived | v1 baseline — Kaggle paths, XGB only |
| `train_v2.py` | Completed | 3-model ensemble, EV=0.03 |
| `train_v3.py` | Building | Phase 1 fixes + new features |
| `submission_v2.csv` | Generated | Pred range [0.38, 0.61] |
| `fold_report_v2.csv` | Generated | MAE=0.179, RMSE=0.235, EV=0.030 |
| `submission_v3.csv` | Pending | After v3 training |
| `fold_report_v3.csv` | Pending | After v3 training |
| `CHANGELOG.md` | Active | This file |
| `requirements.txt` | Active | Exact version pins for Python 3.14 |
