import pandas as pd
import numpy as np

df = pd.read_csv("c:/KruthimaOps/data/train.csv")

# Print columns
numeric_cols = df.select_dtypes(include=[np.number]).columns
corrs = df[numeric_cols].corr()["flood_risk_score"].sort_values(ascending=False)
print("Numeric correlations with flood_risk_score:")
print(corrs)

# Let's inspect the target distribution when features are extreme
print("\nLet's see what features are highly correlated with high target values (e.g. > 0.8)")
high_risk_df = df[df["flood_risk_score"] > 0.8]
print(high_risk_df[numeric_cols].mean() - df[numeric_cols].mean())

# Let's see some categorical feature statistics for high risk vs overall
print("\nCategorical columns distribution for high risk (>0.8) vs overall:")
cat_cols = ["flood_occurrence_current_event", "is_good_to_live", "reason_not_good_to_live", "district"]
for col in cat_cols:
    if col in df.columns:
        print(f"\n--- {col} ---")
        overall = df[col].value_counts(normalize=True).head(5)
        high = high_risk_df[col].value_counts(normalize=True).head(5)
        print("Overall:")
        print(overall)
        print("High Risk (>0.8):")
        print(high)
