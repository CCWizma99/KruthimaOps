import pandas as pd
import numpy as np
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

INPUT_FILE = 'C:/KruthimaOps/data/train_v1002_desinventar.csv'

def apply_pseudo_labels():
    print("[LOAD] Reading dataset...")
    df = pd.read_csv(INPUT_FILE)
    
    # Identify non-synthetic rows
    real_mask = df['is_synthetic'] != 1
    
    # Identify splits within real data
    train_mask = real_mask & df['record_id'].notna()
    test_mask = real_mask & df['record_id'].isna()
    
    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()
    
    # Drop rows missing coordinates in the train set to build a solid model
    train_df = train_df.dropna(subset=['latitude', 'longitude', 'flood_risk_score']).copy()
    
    print(f"Train Set (Recent): {len(train_df)} rows")
    print(f"Test Set (Historical to be labeled): {len(test_df)} rows")
    
    # Extract features
    features_to_drop = ['record_id', 'district', 'place_name', 'flood_risk_score', 'is_synthetic', 'generation_date']
    use_cols = [c for c in train_df.columns if c not in features_to_drop]
    
    X_train = train_df[use_cols].select_dtypes(include=[np.number]).fillna(-999)
    y_train = train_df['flood_risk_score'].values
    
    X_test = test_df[X_train.columns].select_dtypes(include=[np.number]).fillna(-999)
    
    print(f"[TRAIN] Training XGBoost model to generate pseudo-labels...")
    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    print("[TEST] Predicting pseudo-labels for historical records...")
    preds = model.predict(X_test)
    preds = np.clip(preds, 0.0, 1.0)
    
    # Inject pseudo-labels back into the original dataframe
    # We must match the indices perfectly
    df.loc[test_mask, 'flood_risk_score'] = preds
    
    # Save the updated dataset
    df.to_csv(INPUT_FILE, index=False)
    print(f"[SAVE] Pseudo-labels saved to {INPUT_FILE}")
    
    print("\n[DISTRIBUTION] New Target Distribution for Historical Rows:")
    print(pd.Series(preds).describe())

if __name__ == "__main__":
    apply_pseudo_labels()
