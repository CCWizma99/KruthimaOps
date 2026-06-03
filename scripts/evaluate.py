"""
ML Opsidian: Local Competition Scorer
======================================
Replicates the competition's scoring metric locally using OOF predictions.

# REVERSE-ENGINEERED FORMULA (10 data points):
#   LB = (0.392696 * MAE + 0.875527 * RMSE) * (1.0 + 0.406963 * (1 - EV))
# 
# Validated against 10 known LB submissions with MaxErr = 0.00101.

USAGE:
  # Score a specific version's OOF predictions:
  python scripts/evaluate.py --version v19

  # Score with custom OOF file:
  python scripts/evaluate.py --oof submissions/oof_v19.npy

  # Compare all available versions:
  python scripts/evaluate.py --all

  # Score from local metrics directly:
  python scripts/evaluate.py --mae 0.17937 --rmse 0.23500 --ev 0.03060
"""

import argparse
import numpy as np
import pandas as pd
import os
import glob
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, explained_variance_score

# ============================================================
# THE REVERSE-ENGINEERED COMPETITION METRIC
# ============================================================

def competition_score(mae, rmse, ev):
    """
    Compute estimated leaderboard score from MAE, RMSE, EV.
    
    Derived from 21 known LB submissions via least-squares fitting of multiplicative form:
      LB = (0.539328 * MAE + 1.152263 * RMSE) * (1.0 + 0.048467 * (1.0 - EV))
    """
    return (0.539328 * mae + 1.152263 * rmse) * (1.0 + 0.048467 * (1.0 - ev))



def evaluate_predictions(y_true, y_pred, label=""):
    """Full evaluation of predictions against ground truth."""
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    ev   = explained_variance_score(y_true, y_pred)
    lb   = competition_score(mae, rmse, ev)
    
    pred_mean = np.mean(y_pred)
    pred_std  = np.std(y_pred)
    pred_min  = np.min(y_pred)
    pred_max  = np.max(y_pred)
    
    return {
        "label": label,
        "MAE": mae,
        "RMSE": rmse,
        "EV": ev,
        "est_LB": lb,
        "pred_mean": pred_mean,
        "pred_std": pred_std,
        "pred_min": pred_min,
        "pred_max": pred_max,
        "n_samples": len(y_true),
    }


def print_evaluation(result, verbose=True):
    """Pretty-print evaluation results."""
    print(f"\n{'=' * 60}")
    print(f"  COMPETITION SCORE ESTIMATE: {result['label']}")
    print(f"{'=' * 60}")
    print(f"  MAE            : {result['MAE']:.5f}")
    print(f"  RMSE           : {result['RMSE']:.5f}")
    print(f"  Explained Var. : {result['EV']:.5f}")
    print(f"  ---")
    print(f"  Est. LB Score  : {result['est_LB']:.5f}")
    print(f"{'=' * 60}")
    
    if verbose:
        # Component breakdown
        base_err = 0.539328 * result['MAE'] + 1.152263 * result['RMSE']
        penalty = 1.0 + 0.048467 * (1.0 - result['EV'])
        
        print(f"\n  [COMPONENT BREAKDOWN]")
        print(f"    Base Error (0.539*MAE + 1.152*RMSE)        : {base_err:.5f}")
        print(f"    Variance Penalty Factor (1 + 0.048*(1-EV)) : {penalty:.5f}")
        print(f"    Total (Base Error * Penalty)               : {result['est_LB']:.5f}")

        
        print(f"\n  [PREDICTION STATS]")
        print(f"    Mean : {result['pred_mean']:.5f}")
        print(f"    Std  : {result['pred_std']:.5f}")
        print(f"    Range: [{result['pred_min']:.4f}, {result['pred_max']:.4f}]")
        
        # Comparison to known submissions
        print(f"\n  [COMPARISON TO KNOWN LB SCORES]")
        known = [
            ("Rank 1 target",  0.38037, None),
            ("v70_optimized",  0.38216, 0.38216),
            ("v67_optimized",  0.38216, 0.38216),
            ("v42_optimized",  0.38245, 0.38245),
            ("v64_optimized",  0.38256, 0.38256),
            ("v45_optimized",  0.38272, 0.38272),
            ("v44_optimized",  0.38278, 0.38278),
            ("v30",            0.38293, 0.38293),
            ("v33",            0.38294, 0.38294),
            ("v60_optimized",  0.38295, 0.38295),
            ("v38_optimized",  0.38298, 0.38298),
            ("v63_optimized",  0.38309, 0.38309),
            ("v37_optimized",  0.38328, 0.38328),
            ("v20",            0.38331, 0.38331),
            ("v37",            0.38335, 0.38335),
            ("v54_optimized",  0.38337, 0.38337),
            ("v19",            0.38401, 0.38401),
            ("v23 (overfit)",  0.38411, 0.38411),
            ("v13",            0.38476, 0.38476),
            ("v28_kaggle",     0.38499, 0.38499),
            ("v17",            0.38506, 0.38506),
            ("v3",             0.38559, 0.38559),
            ("v10",            0.38598, 0.38598),
            ("v11",            0.38637, 0.38637),
            ("v10_probe_k3.5", 0.41264, 0.41264),
        ]
        for name, lb_actual, _ in known:
            delta = result['est_LB'] - lb_actual
            status = "BETTER" if delta < 0 else "WORSE" if delta > 0 else "EQUAL"
            arrow = "<" if delta < 0 else ">" if delta > 0 else "="
            print(f"    vs {name:<16}: {delta:+.5f} ({status})")
        
        # What would it take to reach rank 1?
        gap = result['est_LB'] - 0.38037
        if gap > 0:
            d_mae = 0.539328 * (1.0 + 0.048467 * (1.0 - result['EV']))
            d_rmse = 1.152263 * (1.0 + 0.048467 * (1.0 - result['EV']))
            d_ev = 0.048467 * (0.539328 * result['MAE'] + 1.152263 * result['RMSE'])
            
            print(f"\n  [GAP TO RANK 1: {gap:.5f}]")
            print(f"    Need MAE  decrease: {gap/d_mae:.5f} ({gap/d_mae/result['MAE']*100:.2f}%)")
            print(f"    Need RMSE decrease: {gap/d_rmse:.5f} ({gap/d_rmse/result['RMSE']*100:.2f}%)")
            print(f"    Need EV   increase: {gap/d_ev:.5f} ({gap/d_ev/result['EV']*100:.1f}%)")
        else:
            print(f"\n  ** ESTIMATED TO BEAT RANK 1 by {-gap:.5f}! **")


def load_training_labels():
    """Load the training target values."""
    train_df = pd.read_csv("data/train.csv")
    train_df = train_df.drop_duplicates()
    return train_df['flood_risk_score'].values


def find_oof_file(version):
    """Find OOF prediction file for a given version."""
    patterns = [
        f"submissions/oof_{version}.npy",
        f"submissions/oof_preds_{version}.npy",
    ]
    for p in patterns:
        if os.path.exists(p):
            return p
    return None


def scan_all_versions():
    """Find all available OOF prediction files."""
    files = glob.glob("submissions/oof_*.npy")
    versions = []
    for f in files:
        basename = os.path.basename(f)
        ver = basename.replace("oof_", "").replace(".npy", "")
        versions.append((ver, f))
    return sorted(versions)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ML Opsidian Local Competition Scorer")
    parser.add_argument("--version", type=str, help="Version to evaluate (e.g., v19)")
    parser.add_argument("--oof", type=str, help="Path to OOF predictions .npy file")
    parser.add_argument("--all", action="store_true", help="Compare all available OOF files")
    parser.add_argument("--mae", type=float, help="Direct MAE value")
    parser.add_argument("--rmse", type=float, help="Direct RMSE value")
    parser.add_argument("--ev", type=float, help="Direct EV value")
    args = parser.parse_args()
    
    # Mode 1: Direct metric input
    if args.mae is not None and args.rmse is not None and args.ev is not None:
        lb = competition_score(args.mae, args.rmse, args.ev)
        result = {
            "label": "Direct Input",
            "MAE": args.mae,
            "RMSE": args.rmse,
            "EV": args.ev,
            "est_LB": lb,
            "pred_mean": 0, "pred_std": 0, "pred_min": 0, "pred_max": 0,
            "n_samples": 0,
        }
        print_evaluation(result, verbose=True)
    
    # Mode 2: Score a specific version
    elif args.version:
        oof_file = find_oof_file(args.version)
        if oof_file is None:
            print(f"[ERROR] No OOF file found for {args.version}")
            print(f"  Training scripts need to save: submissions/oof_{args.version}.npy")
            print(f"\n  Available OOF files:")
            for ver, path in scan_all_versions():
                print(f"    {ver}: {path}")
        else:
            y_true = load_training_labels()
            y_pred = np.load(oof_file)
            result = evaluate_predictions(y_true, y_pred, label=args.version)
            print_evaluation(result, verbose=True)
    
    # Mode 3: Score a custom OOF file
    elif args.oof:
        if not os.path.exists(args.oof):
            print(f"[ERROR] File not found: {args.oof}")
        else:
            y_true = load_training_labels()
            y_pred = np.load(args.oof)
            result = evaluate_predictions(y_true, y_pred, label=args.oof)
            print_evaluation(result, verbose=True)
    
    # Mode 4: Compare all versions
    elif args.all:
        versions = scan_all_versions()
        if not versions:
            print("[ERROR] No OOF files found in submissions/")
            print("  Training scripts need to save OOF predictions as:")
            print("  np.save('submissions/oof_vXX.npy', oof_predictions)")
        else:
            y_true = load_training_labels()
            
            print("=" * 80)
            print("  ALL VERSIONS - COMPETITION SCORE COMPARISON")
            print("=" * 80)
            
            known_lb = {
                "v3": 0.38559, "v10": 0.38598, "v11": 0.38637,
                "v10_probe_k3.5": 0.41264,
                "v13": 0.38476, "v17": 0.38506, "v19": 0.38401,
                "v20": 0.38331, "v23": 0.38411, "v28_kaggle": 0.38499,
                "v30": 0.38293, "v33": 0.38294,
                "v37": 0.38335, "v37_optimized": 0.38328,
                "v38_optimized": 0.38298, "v42_optimized": 0.38245,
                "v44_optimized": 0.38278, "v45_optimized": 0.38272,
                "v54_optimized": 0.38337, "v60_optimized": 0.38295,
                "v63_optimized": 0.38309, "v64_optimized": 0.38256,
                "v67_optimized": 0.38216, "v70_optimized": 0.38216
            }

            
            results = []
            for ver, path in versions:
                y_pred = np.load(path)
                if len(y_pred) != len(y_true):
                    print(f"  [SKIP] {ver}: length mismatch ({len(y_pred)} vs {len(y_true)})")
                    continue
                r = evaluate_predictions(y_true, y_pred, label=ver)
                r["actual_LB"] = known_lb.get(ver, None)
                results.append(r)
            
            # Sort by estimated LB (lower is better)
            results.sort(key=lambda x: x['est_LB'])
            
            print(f"\n  {'Ver':<10} {'MAE':>8} {'RMSE':>8} {'EV':>8} {'est_LB':>8} {'act_LB':>8} {'err':>8}")
            print(f"  {'-'*66}")
            for r in results:
                act = f"{r['actual_LB']:.5f}" if r['actual_LB'] else "    -"
                err = ""
                if r['actual_LB']:
                    err = f"{r['est_LB'] - r['actual_LB']:+.5f}"
                print(f"  {r['label']:<10} {r['MAE']:>8.5f} {r['RMSE']:>8.5f} {r['EV']:>8.5f} {r['est_LB']:>8.5f} {act:>8} {err:>8}")
            
            print(f"\n  Rank 1 target: 0.38037")
    
    # Mode 5: Quick score from known versions (hardcoded)
    else:
        print("ML Opsidian Competition Scorer")
        print("=" * 60)
        print("\nKnown submissions and estimated scores:")
        
        known = [
            ("v3",         0.17962, 0.23520, 0.02889, 0.38559),
            ("v10",        0.17971, 0.23526, 0.02845, 0.38598),
            ("v10_probe_k3.5", 0.19298, 0.24980, -0.09534, 0.41264),
            ("v11",        0.17984, 0.23539, 0.02737, 0.38637),
            ("v13",        0.17937, 0.23500, 0.03060, 0.38476),
            ("v17",        0.17882, 0.23465, 0.03390, 0.38506),
            ("v19",        0.17891, 0.23461, 0.03379, 0.38401),
            ("v20",        0.17865, 0.23439, 0.03564, 0.38331),
            ("v23",        0.17880, 0.23449, 0.03475, 0.38411),
            ("v28_kaggle", 0.17929, 0.23479, 0.03237, 0.38499),
            ("v30",        0.17862, 0.23436, 0.03587, 0.38293),
            ("v33",        0.17863, 0.23444, 0.03519, 0.38294),
            ("v37",        0.17853, 0.23439, 0.03566, 0.38335),
            ("v37_optimized", 0.17851, 0.23436, 0.03589, 0.38328),
            ("v38_optimized", 0.17853, 0.23438, 0.03571, 0.38298),
            ("v42_optimized",  0.17811, 0.23401, 0.03873, 0.38245),
            ("v44",            0.17808, 0.23400, 0.03892, None),
            ("v44_optimized",  0.17806, 0.23399, 0.03892, 0.38278),
            ("v45",            0.17814, 0.23403, 0.03863, None),
            ("v45_optimized",  0.17811, 0.23402, 0.03869, 0.38272),
            ("v54_optimized",  0.17820, 0.23420, 0.03751, 0.38337),
            ("v60_optimized",  0.17827, 0.23413, 0.03782, 0.38295),
            ("v63_optimized",  0.17822, 0.23408, 0.03822, 0.38309),
            ("v64_optimized",  0.17803, 0.23393, 0.03942, 0.38256),
            ("v67_optimized",  0.17803, 0.23396, 0.03919, 0.38216),
        ]

        
        print(f"\n  {'Ver':<10} {'MAE':>8} {'RMSE':>8} {'EV':>8} {'est_LB':>8} {'act_LB':>8}")
        print(f"  {'-'*60}")
        for name, mae, rmse, ev, act_lb in known:
            est = competition_score(mae, rmse, ev)
            act = f"{act_lb:.5f}" if act_lb else "    -"
            print(f"  {name:<10} {mae:>8.5f} {rmse:>8.5f} {ev:>8.5f} {est:>8.5f} {act:>8}")
        
        print(f"\n  Usage:")
        print(f"    python scripts/evaluate.py --version v19")
        print(f"    python scripts/evaluate.py --mae 0.179 --rmse 0.235 --ev 0.03")
        print(f"    python scripts/evaluate.py --all")
