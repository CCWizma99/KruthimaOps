import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

ID_COL = 'record_id'
TARGET_COL = 'flood_risk_score'

# Conservative downstream / risky columns that should be avoided in 'safe' features
RISKY_DOWNSTREAM = [
    'flood_occurrence_current_event',
    'inundation_area_sqm',
    'is_good_to_live',
    'reason_not_good_to_live',
]


def _parse_date(df: pd.DataFrame, col: str = 'generation_date') -> pd.DataFrame:
    if col not in df.columns:
        return df
    ds = pd.to_datetime(df[col], errors='coerce')
    df[col + '_year'] = ds.dt.year.fillna(0).astype(int)
    df[col + '_month'] = ds.dt.month.fillna(0).astype(int)
    df[col + '_day'] = ds.dt.day.fillna(0).astype(int)
    df[col + '_dayofweek'] = ds.dt.dayofweek.fillna(-1).astype(int)

    # simple meteorological seasons (NH-based) — mapping month to season
    season_map = {12: 'winter', 1: 'winter', 2: 'winter',
                  3: 'spring', 4: 'spring', 5: 'spring',
                  6: 'summer', 7: 'summer', 8: 'summer',
                  9: 'autumn', 10: 'autumn', 11: 'autumn'}
    df[col + '_season'] = ds.dt.month.map(season_map).fillna('unknown')
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ported, modular feature engineering from v20/v21 scripts.

    Expects a combined dataframe (train + test) so group statistics are identical.
    This function adds many derived features used by the original v20/v21 pipelines.
    It is safe to call multiple times; missing source columns are ignored.
    """
    df = df.copy()

    def safe_log1p(series: pd.Series) -> pd.Series:
        return np.log1p(pd.to_numeric(series, errors='coerce').fillna(0.0).clip(lower=0.0))

    # helper maps computed on the (combined) dataframe
    if 'district' in df.columns:
        district_elev_std = df.groupby('district')['elevation_m'].std().to_dict()
    else:
        district_elev_std = {}

    if 'landcover' in df.columns and 'inundation_area_sqm' in df.columns:
        landcover_mean_inundation = df.groupby('landcover')['inundation_area_sqm'].mean().to_dict()
    else:
        landcover_mean_inundation = {}

    # safe guards and convenience columns
    # numeric log transforms if originals exist
    if 'distance_to_river_m' in df.columns:
        df['distance_to_river_m_log1p'] = safe_log1p(df['distance_to_river_m'])
    if 'population_density_per_km2' in df.columns:
        df['population_density_per_km2_log1p'] = safe_log1p(df['population_density_per_km2'])
    if 'rainfall_7d_mm' in df.columns:
        df['rainfall_7d_mm_log1p'] = safe_log1p(df['rainfall_7d_mm'])
    if 'monthly_rainfall_mm' in df.columns:
        df['monthly_rainfall_mm_log1p'] = safe_log1p(df['monthly_rainfall_mm'])
    if 'nearest_hospital_km' in df.columns:
        df['nearest_hospital_km_log1p'] = safe_log1p(df['nearest_hospital_km'])
    if 'nearest_evac_km' in df.columns:
        df['nearest_evac_km_log1p'] = safe_log1p(df['nearest_evac_km'])
    if 'inundation_area_sqm' in df.columns:
        df['inundation_area_log'] = safe_log1p(df['inundation_area_sqm'])

    # date-derived fields (if generation_date exists)
    if 'generation_date' in df.columns:
        ds = pd.to_datetime(df['generation_date'], errors='coerce')
        df['generation_date_month'] = ds.dt.month.fillna(0).astype(int)
        df['generation_date_day'] = ds.dt.day.fillna(0).astype(int)
        df['generation_date_dayofweek'] = ds.dt.dayofweek.fillna(-1).astype(int)
        df['generation_date_year'] = ds.dt.year.fillna(0).astype(int)

    # engineered interactions
    # pseudo TWI: proxy for river proximity vs elevation
    if 'distance_to_river_m' in df.columns and 'elevation_m' in df.columns:
        df['pseudo_twi'] = safe_log1p((df['distance_to_river_m'].fillna(0.0) + 1.0) / (df['elevation_m'].clip(lower=0.0).fillna(0.0) + 1.0))

    # river x rain interactions
    if 'distance_to_river_m_log1p' in df.columns and 'rainfall_7d_mm_log1p' in df.columns:
        df['river_rain_interaction'] = df['distance_to_river_m_log1p'] * df['rainfall_7d_mm_log1p']
    if 'distance_to_river_m_log1p' in df.columns and 'monthly_rainfall_mm_log1p' in df.columns:
        df['river_monthly_exposure'] = df['distance_to_river_m_log1p'] * df['monthly_rainfall_mm_log1p']

    # inundation density risk
    if 'inundation_area_log' in df.columns and 'population_density_per_km2_log1p' in df.columns:
        df['inundation_density_risk'] = df['inundation_area_log'] / (df['population_density_per_km2_log1p'].replace(0, np.nan).fillna(1.0) + 1e-6)

    # elevation divergence if yeojohnson processed elevation exists
    if 'elevation_m' in df.columns:
        # if a transformed elevation exists, compute divergence, else use zero
        if 'elevation_m_yeojohnson' in df.columns:
            df['elevation_divergence'] = df['elevation_m'].fillna(0.0) - df['elevation_m_yeojohnson'].fillna(0.0)
        else:
            df['elevation_divergence'] = 0.0

    # infrastructure / rainfall / terrain interactions
    if 'infrastructure_score' in df.columns and 'population_density_per_km2_log1p' in df.columns:
        df['infra_resilience'] = df['infrastructure_score'].fillna(0.0) / (df['population_density_per_km2_log1p'].replace(0, np.nan).fillna(1.0) + 1e-6)
    if 'terrain_roughness_index' in df.columns and 'ndvi_qmap' in df.columns:
        df['terrain_veg_risk'] = df['terrain_roughness_index'].fillna(0.0) * (1.0 - df['ndvi_qmap'].fillna(0.0).clip(-1, 1))

    # evacuation / isolation proxies
    if 'nearest_hospital_km_log1p' in df.columns and 'nearest_evac_km_log1p' in df.columns:
        df['evacuation_difficulty'] = df['nearest_hospital_km_log1p'] + df['nearest_evac_km_log1p']

    # landcover related
    if 'landcover' in df.columns and 'inundation_area_sqm' in df.columns:
        df['landcover_mean_inundation_val'] = df['landcover'].astype(str).map(landcover_mean_inundation).fillna(df['inundation_area_sqm'].mean())
        df['inundation_ratio'] = df['inundation_area_sqm'].fillna(0.0) / (df['landcover_mean_inundation_val'].replace(0, np.nan).fillna(1.0) + 1.0)

    # create coarse grid id as v20 did
    if 'latitude' in df.columns and 'longitude' in df.columns and 'grid_id' not in df.columns:
        lat = df['latitude'].fillna(df['latitude'].median())
        lon = df['longitude'].fillna(df['longitude'].median())
        df['lat_bin'] = (lat / 0.5).astype(int)
        df['lon_bin'] = (lon / 0.5).astype(int)
        df['grid_id'] = df['lat_bin'].astype(str) + '_' + df['lon_bin'].astype(str)

    # drop intermediary helper columns not intended as final features if present
    # keep engineered outputs but remove raw helper columns that duplicate information
    # (we keep many raw columns to preserve compatibility; do not drop record id or target)
    return df


def get_column_groups(df: pd.DataFrame) -> Dict[str, List[str]]:
    cols = set(df.columns.tolist())

    id_cols = [c for c in [ID_COL, 'district', 'place_name'] if c in cols]
    target_cols = [TARGET_COL] if TARGET_COL in cols else []

    # known numeric-ish columns from prompt (fall back to inference)
    known_num = [
        'elevation_m', 'distance_to_river_m', 'population_density_per_km2',
        'built_up_percent', 'rainfall_7d_mm', 'monthly_rainfall_mm',
        'drainage_index', 'ndvi', 'ndwi', 'historical_flood_count',
        'nearest_hospital_km', 'nearest_evac_km', 'inundation_area_sqm',
        'infrastructure_score', 'terrain_roughness_index',
        # pre-engineered
        'distance_to_river_m_log1p', 'population_density_per_km2_log1p',
        'rainfall_7d_mm_log1p', 'monthly_rainfall_mm_log1p',
        'nearest_hospital_km_log1p', 'nearest_evac_km_log1p',
        'elevation_m_yeojohnson', 'drainage_index_yeojohnson',
        'ndvi_qmap', 'ndwi_qmap', 'built_up_percent_qmap',
        'seasonal_index', 'socioeconomic_status_index', 'extreme_weather_index'
    ]

    numeric = [c for c in known_num if c in cols]

    # anything not id/target/numeric and not obviously boolean/int is categorical
    other = [c for c in df.columns if c not in id_cols + target_cols + numeric]
    categorical = [c for c in other if df[c].dtype == object or df[c].dtype.name == 'category']
    # include booleans and small-integer enumerations as categorical
    for c in other:
        if c in categorical:
            continue
        if c in id_cols + target_cols:
            continue
        if pd.api.types.is_bool_dtype(df[c]) or (pd.api.types.is_integer_dtype(df[c]) and df[c].nunique() < 30):
            categorical.append(c)
        elif c not in numeric and c not in categorical:
            # remaining numeric-like columns
            if pd.api.types.is_numeric_dtype(df[c]):
                numeric.append(c)
            else:
                categorical.append(c)

    # remove id/target accidental inclusions
    numeric = [c for c in numeric if c not in id_cols + target_cols]
    categorical = [c for c in categorical if c not in id_cols + target_cols]

    risky = [c for c in RISKY_DOWNSTREAM if c in cols]

    return {
        'id': id_cols,
        'target': target_cols,
        'numeric': numeric,
        'categorical': categorical,
        'risky': risky,
    }


def _label_encode_column(series: pd.Series, mapping: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
    if mapping is None:
        uniques = pd.Series(series.dropna().unique()).astype(str).tolist()
        mapping = {v: i + 1 for i, v in enumerate(uniques)}
    arr = series.astype(object).map(lambda x: mapping.get(str(x), -1)).astype(int).to_numpy()
    return arr, mapping


def build_features(
    df: pd.DataFrame,
    use_safe: bool = True,
    encode_for_tree: bool = True,
) -> Tuple[pd.DataFrame, List[str], List[str], Dict[str, Dict]]:
    """
    Build feature matrix from raw dataframe.

    Returns: (X_df, numeric_cols, categorical_cols, encoders)
    encoders contains label mapping dicts for categorical columns when encode_for_tree=True
    """
    df = df.copy()
    # apply consolidated feature engineering (adds derived features for both train & test)
    df = engineer_features(df)
    # ensure date parsing for downstream logic
    df = _parse_date(df, 'generation_date')

    groups = get_column_groups(df)
    id_cols = groups['id']
    numeric = groups['numeric']
    categorical = groups['categorical']
    risky = groups['risky']

    # choose safe vs full
    if use_safe:
        # exclude risky/downstream columns
        categorical = [c for c in categorical if c not in risky and c not in id_cols]
        numeric = [c for c in numeric if c not in risky and c not in id_cols]
    else:
        # full features: all shared except id and target
        categorical = [c for c in categorical if c not in id_cols]
        numeric = [c for c in numeric if c not in id_cols]

    # basic imputation
    for c in numeric:
        if c not in df.columns:
            continue
        median = df[c].median(skipna=True)
        df[c] = df[c].fillna(median)

    for c in categorical:
        if c not in df.columns:
            continue
        df[c] = df[c].fillna('__MISSING__')

    encoders = {}
    if encode_for_tree:
        # label-encode categoricals to integers; unseen -> -1
        for c in categorical:
            arr, mapping = _label_encode_column(df[c], None)
            df[c] = arr
            encoders[c] = mapping

    feature_cols = numeric + categorical
    X = df[feature_cols].copy()
    return X, numeric, categorical, encoders


def make_feature_sets(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Return two sets of feature names: safe and full (both exclude `record_id` and target)."""
    groups = get_column_groups(df)
    id_cols = groups['id']
    numeric = groups['numeric']
    categorical = groups['categorical']
    risky = groups['risky']

    safe = [c for c in numeric + categorical if c not in risky and c not in id_cols and c != TARGET_COL]
    full = [c for c in numeric + categorical if c not in id_cols and c != TARGET_COL]
    return {'safe': safe, 'full': full}


if __name__ == '__main__':
    print('features module - helpers for building feature matrices')
