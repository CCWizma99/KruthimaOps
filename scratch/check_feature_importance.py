import pandas as pd
import numpy as np
from catboost import CatBoostRegressor

df = pd.read_csv("c:/KruthimaOps/data/train.csv").dropna(subset=["district", "flood_risk_score"]).copy()

TARGET = "flood_risk_score"
ID_COL = "record_id"
DROP_COLS = [ID_COL, "place_name", "is_synthetic", "generation_date", TARGET]

features = [c for c in df.columns if c not in DROP_COLS and not c.endswith("_log1p") and not c.endswith("_yeojohnson") and not c.endswith("_qmap")]
cat_cols = ["district", "landcover", "soil_type", "water_supply", "electricity", "road_quality", "urban_rural", "water_presence_flag", "flood_occurrence_current_event", "is_good_to_live", "reason_not_good_to_live"]

# fill nans
for col in features:
    if col in cat_cols:
        df[col] = df[col].fillna("missing").astype(str)
    else:
        df[col] = df[col].fillna(df[col].median())

X = df[features]
y = df[TARGET]

model = CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6, cat_features=cat_cols, verbose=100)
model.fit(X, y)

imp = pd.DataFrame({
    "feature": features,
    "importance": model.get_feature_importance()
}).sort_values("importance", ascending=False)

print("\nFeature Importances:")
print(imp.head(20))
