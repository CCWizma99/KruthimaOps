import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, explained_variance_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

INPUT_FILE = 'C:/KruthimaOps/data/train_v1002_desinventar.csv'

def evaluate_model():
    print("[LOAD] Reading dataset...")
    df = pd.read_csv(INPUT_FILE)
    
    # Filter for Ground Truth
    df = df[df['is_synthetic'] != 1].copy()
    
    # Identify splits
    train_df = df[df['record_id'].notna()].copy()
    test_df = df[df['record_id'].isna()].copy()
    
    # Standard drops
    train_df = train_df.dropna(subset=['latitude', 'longitude', 'flood_risk_score']).copy()
    test_df = test_df.dropna(subset=['latitude', 'longitude', 'flood_risk_score']).copy()
    
    print(f"Train Set (Recent): {len(train_df)} rows")
    print(f"Test Set (Historical): {len(test_df)} rows")
    
    # Extract features and targets
    features_to_drop = ['record_id', 'district', 'place_name', 'flood_risk_score', 'is_synthetic', 'generation_date']
    use_cols = [c for c in train_df.columns if c not in features_to_drop]
    
    X_train = train_df[use_cols].select_dtypes(include=[np.number]).fillna(-999)
    y_train = train_df['flood_risk_score'].values
    
    X_test = test_df[X_train.columns].fillna(-999)
    y_test = test_df['flood_risk_score'].values
    
    print(f"[TRAIN] Training XGBoost baseline on {len(X_train.columns)} features...")
    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    print("[TEST] Predicting historical records...")
    preds = model.predict(X_test)
    preds = np.clip(preds, 0.0, 1.0)
    
    mae = mean_absolute_error(y_test, preds)
    rmse = root_mean_squared_error(y_test, preds)
    ev = explained_variance_score(y_test, preds)
    
    print("\n==============================================")
    print("  VALIDATION: 800 Modern vs 1500 Historical")
    print("==============================================")
    print(f"  MAE  (Mean Abs Error)      : {mae:.4f}")
    print(f"  RMSE (Root Mean Sq Error)  : {rmse:.4f}")
    print(f"  EV   (Explained Variance)  : {ev:.4f}")
    print("==============================================\n")
    
    print("Target True Distribution:")
    print(pd.Series(y_test).describe())
    print("\nTarget Pred Distribution:")
    print(pd.Series(preds).describe())

if __name__ == "__main__":
    evaluate_model()
