import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, explained_variance_score

df = pd.read_csv("c:/KruthimaOps/data/train.csv").dropna(subset=["district", "flood_risk_score"]).copy()

TARGET = "flood_risk_score"
ID_COL = "record_id"
DROP_COLS = [ID_COL, "place_name", "is_synthetic", "generation_date", TARGET]

# raw numeric and categorical features
features = [c for c in df.columns if c not in DROP_COLS and not c.endswith("_log1p") and not c.endswith("_yeojohnson") and not c.endswith("_qmap")]
cat_cols = ["district", "landcover", "soil_type", "water_supply", "electricity", "road_quality", "urban_rural", "water_presence_flag", "flood_occurrence_current_event", "is_good_to_live", "reason_not_good_to_live"]

# convert categories to 'category' type
for col in features:
    if col in cat_cols:
        df[col] = df[col].fillna("missing").astype("category")
    else:
        df[col] = df[col].fillna(df[col].median())

X = df[features]
y = df[TARGET]

kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(df))

for fold, (tr_idx, va_idx) in enumerate(kf.split(X, y)):
    X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
    X_va, y_va = X.iloc[va_idx], y.iloc[va_idx]
    
    model = LGBMRegressor(n_estimators=1000, learning_rate=0.03, random_state=42, n_jobs=-1, verbosity=-1)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)])
    
    oof[va_idx] = model.predict(X_va)

print("\nBaseline LGBM OOF Stats:")
print(pd.Series(oof).describe())
print("True target stats:")
print(y.describe())
print("OOF MAE:", mean_absolute_error(y, oof))
print("OOF EV:", explained_variance_score(y, oof))
