import pandas as pd
import numpy as np

DI_FILE = 'C:/KruthimaOps/data/SriLankaOldData/DI_report70416.xls'
OUTPUT_FILE = 'C:/KruthimaOps/data/train_v1002_desinventar.csv'

def calculate_true_impact(row):
    score = 0.1  # Baseline risk for any reported event
    
    # 1. Fatalities (+0.4)
    deaths = pd.to_numeric(row.get("Deaths"), errors="coerce")
    deaths = deaths if not pd.isna(deaths) else 0
    if deaths > 0:
        score += min(0.4, (deaths / 5.0) * 0.4)
        
    # 2. Total Destruction (+0.3)
    destroyed = pd.to_numeric(row.get("Houses Destroyed"), errors="coerce")
    destroyed = destroyed if not pd.isna(destroyed) else 0
    if destroyed > 0:
        score += min(0.3, (destroyed / 50.0) * 0.3)
        
    # 3. Partial Damage (+0.2)
    damaged = pd.to_numeric(row.get("Houses Damaged"), errors="coerce")
    damaged = damaged if not pd.isna(damaged) else 0
    if damaged > 0:
        score += min(0.2, (damaged / 500.0) * 0.2)
        
    # 4. Agricultural Loss (+0.1)
    crops = pd.to_numeric(row.get("Damages in crops Ha."), errors="coerce")
    crops = crops if not pd.isna(crops) else 0
    if crops > 0:
        score += min(0.1, (crops / 1000.0) * 0.1)
        
    return min(1.0, score)

def run():
    print("Loading DesInventar TSV...")
    di_df = pd.read_csv(DI_FILE, sep="\t", on_bad_lines='skip')
    di_df.columns = [c.replace('"', '').strip() for c in di_df.columns]
    
    allowed_events = ["FLOOD", "HEAVY RAINS", "CYCLONE", "STORM"]
    di_df = di_df[di_df["Event"].isin(allowed_events)]
    di_df = di_df[di_df["Date (YMD)"].str.startswith("201") | di_df["Date (YMD)"].str.startswith("202")]
    
    if len(di_df) > 1500:
        di_df = di_df.sample(1500, random_state=42)
        
    print(f"Extracted exactly {len(di_df)} rows from TSV.")
    
    # Calculate scores in the exact order
    new_scores = []
    for idx, row in di_df.iterrows():
        new_scores.append(calculate_true_impact(row))
        
    print("Loading CSV dataset...")
    df = pd.read_csv(OUTPUT_FILE)
    
    # Find the rows missing record_id (should be exactly 1500)
    missing_mask = df['record_id'].isna()
    missing_count = missing_mask.sum()
    print(f"Found {missing_count} rows missing record_id in CSV.")
    
    if missing_count == len(new_scores):
        df.loc[missing_mask, 'flood_risk_score'] = new_scores
        print("Scores successfully injected!")
        
        # Save
        df.to_csv(OUTPUT_FILE, index=False)
        print("Saved successfully to", OUTPUT_FILE)
        
        # Quick validation
        print("New distribution of previously flat rows:")
        print(df[missing_mask]['flood_risk_score'].describe())
    else:
        print("Mismatch! Cannot reliably inject.")

if __name__ == "__main__":
    run()
