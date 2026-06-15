import pandas as pd
df = pd.read_csv("c:/KruthimaOps/data/train.csv")

print("Let's look at the average flood_risk_score by rainfall bins:")
df['rainfall_bin'] = pd.qcut(df['rainfall_7d_mm'], 5, duplicates='drop')
print(df.groupby('rainfall_bin', observed=True)['flood_risk_score'].mean())

print("\nLet's look at the average flood_risk_score by inundation_area_sqm bins:")
df['inundation_bin'] = pd.cut(df['inundation_area_sqm'], bins=[-1, 0, 1000, 10000, 50000, 1000000], labels=['0', '0-1k', '1k-10k', '10k-50k', '>50k'])
print(df.groupby('inundation_bin', observed=True)['flood_risk_score'].mean())

print("\nLet's count how many rows have high risk (> 0.7) for each rain bin and inundation bin:")
print(df[df['flood_risk_score'] > 0.7].groupby(['rainfall_bin', 'inundation_bin'], observed=True).size())

print("\nLet's look at the correlation between coordinates and risk:")
print("Latitude correlation:", df['latitude'].corr(df['flood_risk_score']))
print("Longitude correlation:", df['longitude'].corr(df['flood_risk_score']))
