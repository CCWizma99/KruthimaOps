import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import explained_variance_score, mean_squared_error

print("Loading data...")
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

print("\n--- EDGE 1: The 'record_id' Magic ---")
train['id_num'] = train['record_id'].str.extract('(\d+)').astype(float)
test['id_num'] = test['record_id'].str.extract('(\d+)').astype(float)

corr = train['id_num'].corr(train['flood_risk_score'])
print(f"Correlation between record_id number and target: {corr:.4f}")

print("\n--- EDGE 2: Target Distribution ---")
print(train['flood_risk_score'].describe())
print(f"Skewness: {train['flood_risk_score'].skew():.4f}")
print(f"Kurtosis: {train['flood_risk_score'].kurt():.4f}")

print("\n--- EDGE 3: Missing Value Signal ---")
train_orig = pd.read_csv("data/train.csv") # reload to get raw NaNs
missing_cols = train_orig.columns[train_orig.isnull().sum() > 0]
for col in missing_cols:
    train_orig[f'{col}_is_nan'] = train_orig[col].isnull().astype(int)
    corr = train_orig[f'{col}_is_nan'].corr(train_orig['flood_risk_score'])
    print(f"Correlation of {col}_is_nan with target: {corr:.4f}")

print("\n--- EDGE 4: Is_Synthetic Signal ---")
if 'is_synthetic' in train.columns:
    print(train.groupby('is_synthetic')['flood_risk_score'].agg(['mean', 'std', 'count']))

print("\n--- EDGE 5: Generation Date Leakage ---")
if 'generation_date' in train.columns:
    train['date'] = pd.to_datetime(train['generation_date'])
    print(train.groupby(train['date'].dt.date)['flood_risk_score'].agg(['mean', 'count']).head())
    
