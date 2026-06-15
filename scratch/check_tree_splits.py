import pandas as pd
from sklearn.tree import DecisionTreeRegressor, export_text

df = pd.read_csv("c:/KruthimaOps/data/train.csv").dropna(subset=["district", "flood_risk_score"]).copy()

TARGET = "flood_risk_score"
ID_COL = "record_id"
DROP_COLS = [ID_COL, "place_name", "is_synthetic", "generation_date", TARGET]

features = [c for c in df.columns if c not in DROP_COLS and not c.endswith("_log1p") and not c.endswith("_yeojohnson") and not c.endswith("_qmap")]
cat_cols = ["district", "landcover", "soil_type", "water_supply", "electricity", "road_quality", "urban_rural", "water_presence_flag", "flood_occurrence_current_event", "is_good_to_live", "reason_not_good_to_live"]

# convert categories to codes
for col in features:
    if col in cat_cols:
        df[col] = df[col].fillna("missing").astype("category").cat.codes
    else:
        df[col] = df[col].fillna(df[col].median())

X = df[features]
y = df[TARGET]

tree = DecisionTreeRegressor(max_depth=3, random_state=42)
tree.fit(X, y)

print("Decision Tree Rules:")
print(export_text(tree, feature_names=features))
