import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, explained_variance_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import time

print("="*65)
print("  ML Opsidian: Real Physics Predictor (N=802)")
print("="*65)

print("[LOAD] Loading data...")
train_df = pd.read_csv("data/train.csv")
real_df = train_df[train_df["is_synthetic"].isna()].copy()
print(f"   Original shape : {train_df.shape}")
print(f"   Real rows only : {real_df.shape}")

TARGET = "flood_risk_score"

# 1. Purge Downstream Leakage & Synthetic Artifacts
DROP_COLS = [
    "record_id", "place_name", "is_synthetic", "generation_date", TARGET,
    # DOWNSTREAM LEAKAGES (Surveys/Events recorded AFTER the disaster)
    "flood_occurrence_current_event", 
    "is_good_to_live", 
    "reason_not_good_to_live",
    "historical_flood_count" # Arguably a physical feature, but heavily correlated with downstream events. Let's keep it for now.
]

# 2. Define Physical Feature Sets
features = [c for c in real_df.columns if c not in DROP_COLS]
cat_features = [
    "district", "landcover", "soil_type", "water_supply", 
    "electricity", "road_quality", "urban_rural", "water_presence_flag"
]
num_features = [c for c in features if c not in cat_features]

X = real_df[features].copy()
y = real_df[TARGET].copy()

# Simple Imputation
for c in num_features:
    X[c] = X[c].fillna(X[c].median())
for c in cat_features:
    X[c] = X[c].fillna("missing").astype(str)

print(f"\n[FEAT] Physical Predictors: {len(features)} total")
print(f"       ({len(num_features)} Numeric, {len(cat_features)} Categorical)")

# 3. Build Robust Small-Data Pipelines
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)
    ])

# Ridge: Excellent for extracting stable linear physical correlations (e.g. Rainfall up -> Risk up)
ridge_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', Ridge(alpha=10.0))
])

# Random Forest: Constrained to prevent overfitting N=802, looking for physical interactions
rf_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(
        n_estimators=500,
        max_depth=5,
        min_samples_leaf=10,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    ))
])

# 4. 5-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_ridge = np.zeros(len(real_df))
oof_rf    = np.zeros(len(real_df))

print("\n[TRAIN] Running 5-Fold CV on Physical Signals...")
for fold, (tr_idx, va_idx) in enumerate(kf.split(X)):
    X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
    X_va, y_va = X.iloc[va_idx], y.iloc[va_idx]
    
    # Train Ridge
    ridge_model.fit(X_tr, y_tr)
    oof_ridge[va_idx] = ridge_model.predict(X_va)
    
    # Train RF
    rf_model.fit(X_tr, y_tr)
    oof_rf[va_idx] = rf_model.predict(X_va)

# 5. Evaluate Physical Signal Extraction
oof_ens = (oof_ridge + oof_rf) / 2.0

print("\n" + "="*65)
print("  EVALUATION: PHYSICAL SIGNAL STRENGTH")
print("="*65)
print(" [Ridge Regression]")
print(f"   RMSE: {root_mean_squared_error(y, oof_ridge):.4f}")
print(f"   MAE:  {mean_absolute_error(y, oof_ridge):.4f}")
print(f"   EV:   {explained_variance_score(y, oof_ridge):.4f}")

print("\n [Random Forest (Constrained)]")
print(f"   RMSE: {root_mean_squared_error(y, oof_rf):.4f}")
print(f"   MAE:  {mean_absolute_error(y, oof_rf):.4f}")
print(f"   EV:   {explained_variance_score(y, oof_rf):.4f}")

print("\n [Ensemble (Ridge + RF)]")
print(f"   RMSE: {root_mean_squared_error(y, oof_ens):.4f}")
print(f"   MAE:  {mean_absolute_error(y, oof_ens):.4f}")
print(f"   EV:   {explained_variance_score(y, oof_ens):.4f}")
print("="*65)

# 6. Extract Feature Importances (From Ridge + RF)
print("\n[INSIGHTS] What physical features actually drive disaster risk?")

# Ridge Coefficients
feature_names = num_features + list(ridge_model.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out())
ridge_coefs = pd.Series(ridge_model.named_steps['regressor'].coef_, index=feature_names)
print("\n Top 5 Positive Physical Drivers (Linear):")
print(ridge_coefs.sort_values(ascending=False).head(5))
print("\n Top 5 Negative Physical Drivers (Linear):")
print(ridge_coefs.sort_values(ascending=True).head(5))

# Random Forest Importance
rf_importances = pd.Series(rf_model.named_steps['regressor'].feature_importances_, index=feature_names)
print("\n Top 5 Non-Linear Physical Drivers (Random Forest):")
print(rf_importances.sort_values(ascending=False).head(5))
