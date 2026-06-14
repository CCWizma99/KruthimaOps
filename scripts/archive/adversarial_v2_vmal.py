"""
Adversarial Validation Pass #2 — Post-vmal Feature Set
=======================================================
Runs on the CLEAN vmal feature set:
  - vmal's full customized imputation framework applied
  - extreme_weather_index, cyclone_vulnerability, flood_pressure already dropped
  - lat_decimal_len, lon_decimal_len fingerprints included
  - All vmal feature engineering applied

Goal: Find the NEXT tier of covariate-shift contributors beyond extreme_weather_index.
Reports top-20 features by importance. Any feature with importance >50 is a
candidate for adversarial drop in the next pipeline version.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsRegressor
import os
import warnings

warnings.filterwarnings("ignore")

DATA_DIR = "/kaggle/input/competitions/ml-opsidian-genesis-initial-round-26"
if not os.path.exists(DATA_DIR):
    DATA_DIR = "data"

print("=" * 65)
print("  ADVERSARIAL VALIDATION PASS #2 — vmal CLEAN FEATURE SET")
print("=" * 65)

# -----------------------------------------------------------------
# 1. LOAD
# -----------------------------------------------------------------
train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test_df  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
train_df = train_df.drop_duplicates()
print(f"   Train: {train_df.shape}  Test: {test_df.shape}")

# -----------------------------------------------------------------
# 1.3. PRECISION FINGERPRINTS (same as vmal)
# -----------------------------------------------------------------
for df in [train_df, test_df]:
    df['lat_decimal_len'] = df['latitude'].apply(
        lambda x: len(str(x).split('.')[1]) if '.' in str(x) and str(x).lower() != 'nan' else 0
    )
    df['lon_decimal_len'] = df['longitude'].apply(
        lambda x: len(str(x).split('.')[1]) if '.' in str(x) and str(x).lower() != 'nan' else 0
    )

# -----------------------------------------------------------------
# 1.4. VALUE FREQUENCY COUNT FEATURES (same as vmal)
# -----------------------------------------------------------------
FREQ_COLS = [
    'latitude', 'longitude', 'elevation_m', 'distance_to_river_m',
    'rainfall_7d_mm', 'monthly_rainfall_mm', 'inundation_area_sqm'
]
combined_raw = pd.concat([
    train_df[[c for c in FREQ_COLS if c in train_df.columns]],
    test_df[[c for c in FREQ_COLS if c in test_df.columns]]
], ignore_index=True)

for col in FREQ_COLS:
    if col in combined_raw.columns:
        freq_map = combined_raw[col].value_counts().to_dict()
        for df in [train_df, test_df]:
            if col in df.columns:
                df[f'{col}_freq'] = df[col].map(freq_map).fillna(0).astype(float)

# -----------------------------------------------------------------
# 2. ROBUST CUSTOMIZED IMPUTATION FRAMEWORK (identical to vmal)
# -----------------------------------------------------------------
print("\n[IMPUTE] Applying vmal customized imputation framework...")
combined = pd.concat([
    train_df.drop(columns=['flood_risk_score'], errors='ignore'), test_df
], ignore_index=True)

# 2.1. Coordinate Geospatial Hot-Deck
coords_lookup = combined.groupby(['place_name', 'district'])[['latitude', 'longitude']].median().to_dict('index')
for df in [train_df, test_df]:
    mask = df['latitude'].isnull() & df['place_name'].notnull() & df['district'].notnull()
    for idx in df[mask].index:
        key = (df.loc[idx, 'place_name'], df.loc[idx, 'district'])
        if key in coords_lookup and not np.isnan(coords_lookup[key]['latitude']):
            df.loc[idx, 'latitude'] = coords_lookup[key]['latitude']
            df.loc[idx, 'longitude'] = coords_lookup[key]['longitude']

for col in ['latitude', 'longitude']:
    district_median = combined.groupby('district')[col].median().to_dict()
    global_median = train_df[col].median()
    for df in [train_df, test_df]:
        df[col] = df[col].fillna(df['district'].map(district_median))
        df[col] = df[col].fillna(global_median)

# 2.2. KNN for geospatial/distance columns
knn_cols = ['elevation_m', 'distance_to_river_m', 'nearest_hospital_km', 'nearest_evac_km']
for col in knn_cols:
    donor_pool = combined.dropna(subset=['latitude', 'longitude', col])
    knn = KNeighborsRegressor(n_neighbors=3, weights='distance')
    knn.fit(donor_pool[['latitude', 'longitude']], donor_pool[col])
    for df in [train_df, test_df]:
        mm = df[col].isnull()
        if mm.any():
            df.loc[mm, col] = knn.predict(df.loc[mm, ['latitude', 'longitude']])
        district_median = combined.groupby('district')[col].median().to_dict()
        df[col] = df[col].fillna(df['district'].map(district_median))
        df[col] = df[col].fillna(train_df[col].median())

# 2.3. Drainage index
drainage_lookup    = combined.groupby(['soil_type', 'landcover'])['drainage_index'].median().to_dict()
soil_drain_lookup  = combined.groupby('soil_type')['drainage_index'].median().to_dict()
dist_drain_lookup  = combined.groupby('district')['drainage_index'].median().to_dict()
global_drain_med   = train_df['drainage_index'].median()
for df in [train_df, test_df]:
    mask = df['drainage_index'].isnull()
    for idx in df[mask].index:
        soil = df.loc[idx, 'soil_type']; lc = df.loc[idx, 'landcover']; dist = df.loc[idx, 'district']
        val = np.nan
        if pd.notnull(soil) and pd.notnull(lc) and (soil, lc) in drainage_lookup:
            val = drainage_lookup[(soil, lc)]
        if np.isnan(val) and pd.notnull(soil) and soil in soil_drain_lookup:
            val = soil_drain_lookup[soil]
        if np.isnan(val) and pd.notnull(dist) and dist in dist_drain_lookup:
            val = dist_drain_lookup[dist]
        df.loc[idx, 'drainage_index'] = global_drain_med if np.isnan(val) else val

# 2.4. NDVI / NDWI
for col in ['ndvi', 'ndwi']:
    dist_lc_lookup = combined.groupby(['district', 'landcover'])[col].median().to_dict()
    lc_lookup      = combined.groupby('landcover')[col].median().to_dict()
    global_median  = train_df[col].median()
    for df in [train_df, test_df]:
        mask = df[col].isnull()
        for idx in df[mask].index:
            dist = df.loc[idx, 'district']; lc = df.loc[idx, 'landcover']
            val = np.nan
            if pd.notnull(dist) and pd.notnull(lc) and (dist, lc) in dist_lc_lookup:
                val = dist_lc_lookup[(dist, lc)]
            if np.isnan(val) and pd.notnull(lc) and lc in lc_lookup:
                val = lc_lookup[lc]
            df.loc[idx, col] = global_median if np.isnan(val) else val

# 2.5. Human/Infrastructure
dev_cols = ['population_density_per_km2', 'built_up_percent', 'infrastructure_score']
for col in dev_cols:
    dist_ur_lookup = combined.groupby(['district', 'urban_rural'])[col].median().to_dict()
    ur_lookup      = combined.groupby('urban_rural')[col].median().to_dict()
    dist_lookup    = combined.groupby('district')[col].median().to_dict()
    global_median  = train_df[col].median()
    for df in [train_df, test_df]:
        mask = df[col].isnull()
        for idx in df[mask].index:
            dist = df.loc[idx, 'district']; ur = df.loc[idx, 'urban_rural']
            val = np.nan
            if pd.notnull(dist) and pd.notnull(ur) and (dist, ur) in dist_ur_lookup:
                val = dist_ur_lookup[(dist, ur)]
            if np.isnan(val) and pd.notnull(ur) and ur in ur_lookup:
                val = ur_lookup[ur]
            if np.isnan(val) and pd.notnull(dist) and dist in dist_lookup:
                val = dist_lookup[dist]
            df.loc[idx, col] = global_median if np.isnan(val) else val

# 2.6. Categorical via District Mode
cat_impute_cols = [
    "district", "landcover", "soil_type", "water_supply",
    "electricity", "road_quality", "urban_rural", "water_presence_flag"
]
for col in cat_impute_cols:
    dist_modes  = combined.groupby('district')[col].agg(
        lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
    ).to_dict()
    global_mode = combined[col].mode().iloc[0]
    for df in [train_df, test_df]:
        if col == "district":
            df[col] = df[col].fillna(global_mode)
        else:
            df[col] = df[col].fillna(df['district'].map(dist_modes))
            df[col] = df[col].fillna(global_mode)

print("   Imputation complete.")

# -----------------------------------------------------------------
# 3. FEATURE ENGINEERING (identical to vmal — no extreme_weather_index deps)
# -----------------------------------------------------------------
print("\n[FEAT] Engineering features...")
district_elev_std        = combined.groupby('district')['elevation_m'].std().to_dict()
landcover_mean_inundation = combined.groupby('landcover')['inundation_area_sqm'].mean().to_dict()
soil_infilt_map   = {'Sandy': 0.8, 'Loamy': 0.6, 'Silty': 0.4, 'Clay': 0.2, 'Peaty': 0.1}
cyclone_districts = {'Batticaloa', 'Trincomalee', 'Ampara', 'Mullaitivu', 'Jaffna'}
wet_zone_districts = {'Colombo', 'Gampaha', 'Kalutara', 'Galle', 'Matara', 'Ratnapura', 'Kegalle'}

def engineer_features(df):
    df = df.copy()
    df["confirmed_severe_risk"] = (
        (df["flood_occurrence_current_event"].astype(str).str.strip().str.lower() == "yes") &
        (df["is_good_to_live"].astype(str).str.strip().str.lower() == "no") &
        (df["inundation_area_sqm"] > 0)
    ).astype(int)
    df["no_flood_confirmed"] = (
        (df["flood_occurrence_current_event"].astype(str).str.strip().str.lower() == "no") &
        (df["is_good_to_live"].astype(str).str.strip().str.lower() == "yes")
    ).astype(int)
    df["inundation_per_capita"] = df["inundation_area_sqm"] / (np.expm1(df["population_density_per_km2_log1p"]) + 1.0)
    has_reason = (~df["reason_not_good_to_live"].astype(str).str.strip().str.lower().isin(
        ["nan", "none", "", "missing", "n/a"])).astype(int)
    df["downstream_risk_count"] = (
        (df["flood_occurrence_current_event"].astype(str).str.strip().str.lower() == "yes").astype(int) +
        (df["is_good_to_live"].astype(str).str.strip().str.lower() == "no").astype(int) +
        has_reason + (df["inundation_area_sqm"] > 0).astype(int)
    )
    df['downstream_sig'] = (
        df['flood_occurrence_current_event'].astype(str).str.strip() + "_" +
        df['is_good_to_live'].astype(str).str.strip() + "_" +
        df['reason_not_good_to_live'].astype(str).str.strip()
    )
    has_inundation = (df["inundation_area_sqm"] > 0).astype(int)
    df["downstream_quad_sig"] = (
        df["flood_occurrence_current_event"].astype(str).str.strip() + "_" +
        df["is_good_to_live"].astype(str).str.strip() + "_" +
        df["reason_not_good_to_live"].astype(str).str.strip() + "_" +
        has_inundation.astype(str)
    )
    date_series = pd.to_datetime(df['generation_date'])
    df['month'] = date_series.dt.month
    df['is_yala'] = df['month'].isin([5, 6, 7, 8, 9]).astype(int)
    df['is_maha'] = df['month'].isin([11, 12, 1]).astype(int)
    df['zone_code'] = df['district'].astype(str).map(lambda x: 1 if x in wet_zone_districts else 2)
    df['monsoon_impact'] = (df['rainfall_7d_mm'] * df['is_yala'] * (df['zone_code'] == 1).astype(int) +
                            df['rainfall_7d_mm'] * df['is_maha'] * (df['zone_code'] == 2).astype(int))
    df['urban_runoff_potential']  = df['rainfall_7d_mm'] * df['built_up_percent'] * (1.0 / (df['drainage_index'] + 1e-5))
    df['fluvial_risk_score_feat'] = df['rainfall_7d_mm'] * (1.0 / (df['distance_to_river_m'] + 1.0))
    df['soil_infiltration']       = df['soil_type'].astype(str).map(soil_infilt_map).fillna(0.4)
    df['soil_saturation_limit']   = df['rainfall_7d_mm'] / (df['soil_infiltration'] + 0.1)
    df['pseudo_twi']  = np.log1p((df['distance_to_river_m'] + 1.0) / (df['elevation_m'].clip(lower=0.0) + 1.0))
    df['flatness_index']   = df['district'].astype(str).map(district_elev_std).fillna(df['elevation_m'].std())
    df['in_cyclone_path']  = df['district'].astype(str).map(lambda x: 1 if x in cyclone_districts else 0)
    # [ADVERSARIAL DROP] extreme_weather_index, cyclone_vulnerability, flood_pressure removed
    df['slope_proxy']          = df['elevation_m'] / (df['distance_to_river_m'] + 1.0)
    df['isolation_index']      = np.log1p(df['nearest_hospital_km']) + np.log1p(df['nearest_evac_km'])
    df['vulnerability']        = df['isolation_index'] / (df['infrastructure_score'] + 1.0)
    df['elevation_divergence'] = df['elevation_m'] - df['elevation_m_yeojohnson']
    df['infra_deficit_sig']    = (
        df['water_supply'].astype(str).str.strip() + "_" +
        df['electricity'].astype(str).str.strip() + "_" +
        df['road_quality'].astype(str).str.strip()
    )
    df["inundation_area_log"]          = np.log1p(df["inundation_area_sqm"])
    df["flood_occurrence_yes"]         = (df["flood_occurrence_current_event"].astype(str).str.strip().str.lower() == "yes").astype(int)
    df["inundation_flood_interaction"] = df["flood_occurrence_yes"] * df["inundation_area_log"]
    df["river_rain_interaction"]       = df["distance_to_river_m_log1p"] * df["rainfall_7d_mm_log1p"]
    df["river_monthly_exposure"]       = df["distance_to_river_m_log1p"] * df["monthly_rainfall_mm_log1p"]
    df["elev_rain_risk"]               = df["elevation_m_yeojohnson"] / (df["rainfall_7d_mm_log1p"] + 1e-6)
    df["water_signal"]                 = df["ndwi_qmap"].clip(lower=0)
    df["drainage_deficit"]             = (df["rainfall_7d_mm_log1p"] + 1) * (1.0 - df["drainage_index_yeojohnson"].clip(0, 1))
    df["infra_resilience"]             = df["infrastructure_score"] / (df["population_density_per_km2_log1p"] + 1e-6)
    df["evacuation_difficulty"]        = df["nearest_hospital_km_log1p"] + df["nearest_evac_km_log1p"]
    df["inundation_density_risk"]      = df["inundation_area_log"] / (df["population_density_per_km2_log1p"] + 1e-6)
    df["terrain_veg_risk"]             = df["terrain_roughness_index"] * (1.0 - df["ndvi_qmap"].clip(-1, 1))
    df["is_repeat_flood_zone"]         = (df["historical_flood_count"] > 2).astype(int)
    df["rain_spike_ratio"]             = df["rainfall_7d_mm"] / (df["monthly_rainfall_mm"] + 1e-6)
    df["confirmed_risk"]               = (
        (df["flood_occurrence_current_event"].astype(str).str.strip().str.lower() == "yes") &
        (df["is_good_to_live"].astype(str).str.strip().str.lower() == "no")
    ).astype(int)
    df['landcover_mean_inundation_val'] = df['landcover'].astype(str).map(landcover_mean_inundation).fillna(
        combined['inundation_area_sqm'].mean()
    )
    df['inundation_ratio']          = df['inundation_area_sqm'] / (df['landcover_mean_inundation_val'] + 1.0)
    ndwi_clip = df["ndwi_qmap"].clip(lower=0.0)
    ndvi_clip = df["ndvi_qmap"].clip(-1.0, 1.0).clip(lower=0.0)
    df["pooling_vulnerability"]     = ndwi_clip * (1.0 - ndvi_clip)
    df["soil_drainage_saturation"]  = df["soil_saturation_limit"] * (1.0 - df["drainage_index_yeojohnson"].clip(0.0, 1.0))
    lat = df["latitude"].fillna(df["latitude"].median())
    lon = df["longitude"].fillna(df["longitude"].median())
    df["grid_id_100"] = (lat / 1.0).astype(int).astype(str)   + "_" + (lon / 1.0).astype(int).astype(str)
    df["grid_id_050"] = (lat / 0.5).astype(int).astype(str)   + "_" + (lon / 0.5).astype(int).astype(str)
    df["grid_id_025"] = (lat / 0.25).astype(int).astype(str)  + "_" + (lon / 0.25).astype(int).astype(str)
    df["grid_id_012"] = (lat / 0.125).astype(int).astype(str) + "_" + (lon / 0.125).astype(int).astype(str)
    df["lat_bin"] = (lat / 0.5).astype(int)
    df["lon_bin"] = (lon / 0.5).astype(int)
    df["grid_id"] = df["lat_bin"].astype(str) + "_" + df["lon_bin"].astype(str)
    df = df.drop(columns=["inundation_area_sqm", "landcover_mean_inundation_val"])
    return df

train_df = engineer_features(train_df)
test_df  = engineer_features(test_df)

# -----------------------------------------------------------------
# 4. BUILD ADVERSARIAL DATASET
# -----------------------------------------------------------------
DROP_COLS  = ['record_id', 'place_name', 'is_synthetic', 'generation_date',
              'flood_risk_score', 'is_test',
              # Already-dropped adversarial features from vmal
              'extreme_weather_index']

train_df['is_test'] = 0
test_df['is_test']  = 1

combined_adv = pd.concat([
    train_df.drop(columns=['flood_risk_score', 'is_pseudo'], errors='ignore'),
    test_df.drop(columns=['is_pseudo'], errors='ignore')
], ignore_index=True)

# Drop string composites and spatial helpers (not in BASE_FEATURES)
STRING_COMPOSITES = ['downstream_sig', 'downstream_quad_sig', 'infra_deficit_sig',
                     'grid_id', 'grid_id_100', 'grid_id_050', 'grid_id_025', 'grid_id_012']
all_drop = DROP_COLS + STRING_COMPOSITES
features = [c for c in combined_adv.columns if c not in all_drop]

print(f"\n[ADV] Running adversarial classifier on {len(features)} post-vmal features...")

# Encode for LGB — handle object, StringDtype, CategoricalDtype, and numeric columns
for col in features:
    is_str_or_cat = (
        pd.api.types.is_string_dtype(combined_adv[col]) or
        pd.api.types.is_object_dtype(combined_adv[col]) or
        isinstance(combined_adv[col].dtype, pd.CategoricalDtype)
    )
    if is_str_or_cat:
        combined_adv[col] = combined_adv[col].astype(str).fillna("missing")
        combined_adv[col] = combined_adv[col].astype('category').cat.codes
    else:
        combined_adv[col] = pd.to_numeric(combined_adv[col], errors='coerce')
        combined_adv[col] = combined_adv[col].fillna(combined_adv[col].median())

X = combined_adv[features].copy()
y = combined_adv['is_test'].values

# -----------------------------------------------------------------
# 5. 5-FOLD STRATIFIED CV ADVERSARIAL MODEL
# -----------------------------------------------------------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(y))
feature_importances = np.zeros(len(features))

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
    X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
    X_va, y_va = X.iloc[va_idx], y[va_idx]

    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=6,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=2.0,
        random_state=42 + fold,
        n_jobs=-1,
        verbosity=-1
    )
    model.fit(X_tr, y_tr)
    oof_preds[va_idx] = model.predict_proba(X_va)[:, 1]
    feature_importances += model.feature_importances_ / 5.0

# -----------------------------------------------------------------
# 6. REPORT
# -----------------------------------------------------------------
auc = roc_auc_score(y, oof_preds)

print("\n" + "=" * 65)
print(f"  ADVERSARIAL AUC (Pass #2, post extreme_weather_index drop): {auc:.5f}")
print("=" * 65)

if auc > 0.60:
    print("\n[WARNING] Significant residual covariate shift detected.")
    print("  => More features need to be dropped before vmal will generalize cleanly.")
elif auc > 0.52:
    print("\n[NOTE] Mild residual shift. Check the top features for candidates to drop.")
else:
    print("\n[CLEAN] AUC ~0.50 — No meaningful residual shift after adversarial drops!")

imp_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
imp_df = imp_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)

print(f"\n{'Rank':<5} {'Feature':<45} {'Importance':>12}  {'Drop?':>8}")
print("-" * 75)
for i, row in imp_df.head(20).iterrows():
    flag = "  <<< DROP" if row['Importance'] > 50 else ""
    print(f"  {i+1:<4} {row['Feature']:<45} {row['Importance']:>12.1f}{flag}")

print("\n" + "=" * 65)
print(f"  ADVERSARIAL AUC SUMMARY")
print(f"  Pass #1 AUC (raw features):          0.63780  (approx)")
print(f"  Pass #2 AUC (post-vmal drops):       {auc:.5f}")
print(f"  Improvement:                         {0.6378 - auc:+.5f}")
print("=" * 65)

# Save report
imp_df.to_csv("adversarial_v2_importance.csv", index=False)
imp_df.to_csv("submissions/adversarial_v2_importance.csv", index=False)
print(f"\n[DONE] Saved adversarial_v2_importance.csv")
print(f"  AUC = {auc:.5f}")
print(f"  Top shift contributor: {imp_df.iloc[0]['Feature']} (importance={imp_df.iloc[0]['Importance']:.1f})")
print(f"  2nd:  {imp_df.iloc[1]['Feature']} (importance={imp_df.iloc[1]['Importance']:.1f})")
print(f"  3rd:  {imp_df.iloc[2]['Feature']} (importance={imp_df.iloc[2]['Importance']:.1f})")
print(f"  4th:  {imp_df.iloc[3]['Feature']} (importance={imp_df.iloc[3]['Importance']:.1f})")
print(f"  5th:  {imp_df.iloc[4]['Feature']} (importance={imp_df.iloc[4]['Importance']:.1f})")
