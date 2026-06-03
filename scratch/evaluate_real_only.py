import numpy as np
import pandas as pd
import glob
import os
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, explained_variance_score

def competition_score(mae, rmse, ev):
    return (0.392696 * mae + 0.875527 * rmse) * (1.0 + 0.406963 * (1.0 - ev))

def main():
    print("======================================================================")
    print("  ML OPSIDIAN - EVALUATE OOF ON REAL ROWS ONLY")
    print("======================================================================\n")

    # Load targets and synthetic flags
    train_df = pd.read_csv("data/train.csv")
    train_df = train_df.drop_duplicates()
    
    real_mask = train_df['is_synthetic'].isna()
    y_true = train_df['flood_risk_score'].values
    y_real = y_true[real_mask]
    
    print(f"Total rows: {len(train_df)}")
    print(f"Real rows: {len(y_real)}")
    
    # Discover OOF files
    oof_files = glob.glob("submissions/oof_*.npy")
    oof_data = {}
    
    for f in oof_files:
        basename = os.path.basename(f)
        ver = basename.replace("oof_", "").replace(".npy", "")
        pred = np.load(f)
        if len(pred) == len(y_true):
            oof_data[ver] = pred
        else:
            # Maybe it's already real-only
            if len(pred) == len(y_real):
                print(f"   [NOTE] {ver} has exactly {len(y_real)} rows (already real-only).")
                oof_data[ver + "_real"] = pred

    # Add v29 from root if not already in submissions
    if os.path.exists("oof_v29.npy") and "v29" not in oof_data:
        pred = np.load("oof_v29.npy")
        if len(pred) == len(y_true):
            oof_data["v29"] = pred

    # Known actual LB scores
    known_lb = {
        "v3": 0.38559, "v10": 0.38598, "v11": 0.38637,
        "v13": 0.38476, "v17": 0.38506, "v19": 0.38401,
        "v20": 0.38331, "v23": 0.38411, "v28_kaggle": 0.38499,
        "v30": 0.38293, "v33": 0.38294
    }

    results = []
    for ver, pred in oof_data.items():
        if ver.endswith("_real"):
            pred_real = pred
            clean_ver = ver.replace("_real", "")
        else:
            pred_real = pred[real_mask]
            clean_ver = ver
            
        mae = mean_absolute_error(y_real, pred_real)
        rmse = root_mean_squared_error(y_real, pred_real)
        ev = explained_variance_score(y_real, pred_real)
        lb = competition_score(mae, rmse, ev)
        
        results.append({
            "Version": clean_ver,
            "MAE": mae,
            "RMSE": rmse,
            "EV": ev,
            "Est_LB": lb,
            "Actual_LB": known_lb.get(clean_ver, None)
        })
        
    df_results = pd.DataFrame(results).sort_values("Est_LB")
    
    print("\n  Ver             MAE     RMSE       EV   Est_LB   Act_LB      Err")
    print("  " + "-" * 70)
    for idx, row in df_results.iterrows():
        act = f"{row['Actual_LB']:.5f}" if row['Actual_LB'] else "    -"
        err = f"{row['Est_LB'] - row['Actual_LB']:+.5f}" if row['Actual_LB'] else ""
        print(f"  {row['Version']:<12} {row['MAE']:>8.5f} {row['RMSE']:>8.5f} {row['EV']:>8.5f} {row['Est_LB']:>8.5f} {act:>8} {err:>8}")

if __name__ == "__main__":
    main()
