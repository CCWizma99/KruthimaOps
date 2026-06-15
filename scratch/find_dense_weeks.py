import pandas as pd
from datetime import datetime, timedelta

df = pd.read_csv("c:/KruthimaOps/data/train.csv")
df["generation_date"] = pd.to_datetime(df["generation_date"])

# Group by date and count unique districts
daily_districts = df.groupby("generation_date")["district"].nunique()

# Slide a 7-day window
results = []
start_date = df["generation_date"].min()
end_date = df["generation_date"].max()

curr = start_date
while curr <= end_date - timedelta(days=6):
    window_dates = [curr + timedelta(days=i) for i in range(7)]
    sub = df[df["generation_date"].isin(window_dates)]
    
    unique_districts = sub["district"].nunique()
    total_rows = len(sub)
    
    # Check if we have high-risk rows (risk > 0.8) and low-risk rows (risk < 0.2) in this week
    high_risk_count = (sub["flood_risk_score"] > 0.8).sum()
    low_risk_count = (sub["flood_risk_score"] < 0.2).sum()
    
    results.append({
        "start_date": curr.strftime("%Y-%m-%d"),
        "end_date": (curr + timedelta(days=6)).strftime("%Y-%m-%d"),
        "unique_districts": unique_districts,
        "total_rows": total_rows,
        "high_risk_rows": int(high_risk_count),
        "low_risk_rows": int(low_risk_count)
    })
    curr += timedelta(days=1)

res_df = pd.DataFrame(results)
print("Top 10 weeks with highest district coverage:")
print(res_df.sort_values("unique_districts", ascending=False).head(10))
