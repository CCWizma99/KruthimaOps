import sys
sys.path.append('c:/KruthimaOps/production')
import app.inference.v703_engine as eng
import numpy as np
import pandas as pd

eng.load_artifacts()
df = pd.read_csv("c:/KruthimaOps/data/train.csv").dropna(subset=["district"]).copy()

print("Loaded train.csv:", len(df))

# We will run the inference pipeline step by step for the entire training set
# and inspect individual model predictions
preds_list = []
for idx, row in df.head(1000).iterrows(): # first 1000 rows
    payload = {
        "district": str(row["district"]),
        "rainfall_7d_mm": float(row["rainfall_7d_mm"]) if not pd.isna(row["rainfall_7d_mm"]) else 0.0,
        "inundation_area_sqm": float(row["inundation_area_sqm"]) if not pd.isna(row["inundation_area_sqm"]) else 0.0,
        "flood_occurrence_current_event": str(row["flood_occurrence_current_event"]) if not pd.isna(row["flood_occurrence_current_event"]) else "No",
        "is_good_to_live": str(row["is_good_to_live"]) if not pd.isna(row["is_good_to_live"]) else "Yes",
        "reason_not_good_to_live": str(row["reason_not_good_to_live"]) if not pd.isna(row["reason_not_good_to_live"]) else "None"
    }
    
    # Run steps
    district = payload["district"]
    base_row = dict(eng._DISTRICT_REF[district])
    base_row.update(payload)
    base_row["rainfall_7d_mm_log1p"] = float(np.log1p(payload["rainfall_7d_mm"]))
    
    for col in eng._ARTIFACTS["freq_cols"]:
        freq_map = eng._ARTIFACTS["freq_maps"].get(col, {})
        raw_val  = base_row.get(col)
        base_row[f"{col}_freq"] = float(freq_map.get(raw_val, 0))
    
    base_row["inundation_area_sqm"] = payload["inundation_area_sqm"]
    row_df = pd.DataFrame([base_row])
    row_df = eng._engineer_features(row_df)
    row_df = eng._apply_te_maps(row_df)
    
    FEATURES = eng._FEATURE_LISTS["FEATURES"]
    for col in FEATURES:
        if col not in row_df.columns:
            row_df[col] = 0.0
    row_df = row_df[FEATURES]
    for col in row_df.columns:
        if col not in eng._FEATURE_LISTS["cat_feature_names"]:
            if row_df[col].dtype in ["float64", "int64", "float32", "int32"]:
                row_df[col] = row_df[col].fillna(0.0)
                
    X_xgb, X_cat, X_lgb = eng._align_dtypes(row_df)
    
    p_xgb1   = float(eng._XGB1.predict(X_xgb)[0])
    p_cat1   = float(eng._CAT1.predict(X_cat)[0])
    p_cat2   = float(eng._CAT2.predict(X_cat)[0])
    p_catrmse = float(eng._CATRMSE.predict(X_cat)[0])
    p_lgb1   = float(eng._LGB1.predict(X_lgb)[0])
    p_xgb2   = float(eng._XGB2.predict(X_xgb)[0])
    
    stacked = np.dot([p_xgb1, p_cat1, p_cat2, p_catrmse, p_lgb1, p_xgb2], eng._STACKER["weights"]) + eng._STACKER["bias"]
    
    preds_list.append({
        "true": row["flood_risk_score"],
        "xgb1": p_xgb1,
        "cat1": p_cat1,
        "cat2": p_cat2,
        "catrmse": p_catrmse,
        "lgb1": p_lgb1,
        "xgb2": p_xgb2,
        "stacked": stacked
    })

res = pd.DataFrame(preds_list)
print("\nPredictions stats:")
print(res.describe())
