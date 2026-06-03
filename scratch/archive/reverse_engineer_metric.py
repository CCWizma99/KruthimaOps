"""
Reverse-engineer the competition metric from 6 known LB submissions.

We have:
  v3:         LB=0.38559
  v10:        LB=0.38598
  v10_probe:  LB=0.41264 (k=3.5 variance stretch of v10)
  v11:        LB=0.38637
  v13:        LB=0.38476
  v17:        LB=0.38506

Strategy:
  1. Load all submission predictions
  2. Compute prediction statistics for each (mean, std, range, etc.)
  3. Use OOF predictions where available to get local metrics
  4. Try multiple candidate metric formulas
  5. Use scipy.optimize to fit parameters
  6. Identify the formula that best explains all 6 LB scores
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from itertools import combinations

# =============================================================
# 1. LOAD ALL SUBMISSIONS AND COMPUTE PREDICTION STATS
# =============================================================
print("=" * 70)
print("  METRIC REVERSE ENGINEERING")
print("=" * 70)

submissions = {
    "v3":         {"file": "submissions/submission_v3.csv",               "lb": 0.38559},
    "v10":        {"file": "submissions/submission_v10.csv",              "lb": 0.38598},
    "v10_probe":  {"file": "submissions/submission_v10_probe_k3.5.csv",  "lb": 0.41264},
    "v11":        {"file": "submissions/submission_v11.csv",              "lb": 0.38637},
    "v13":        {"file": "submissions/submission_v13.csv",              "lb": 0.38476},
    "v17":        {"file": "submissions/submission_v17.csv",              "lb": 0.38506},
}

# Known local OOF metrics
local_metrics = {
    "v11": {"MAE": 0.17984, "RMSE": 0.23539, "EV": 0.02737},
    "v13": {"MAE": 0.17937, "RMSE": 0.23500, "EV": 0.03060},
    "v17": {"MAE": 0.17882, "RMSE": 0.23465, "EV": 0.03390},
}

# Load fold reports for additional metrics
try:
    fr_v3 = pd.read_csv("submissions/fold_report_v3.csv")
    local_metrics["v3"] = {
        "MAE": fr_v3["MAE"].mean(),
        "RMSE": fr_v3["RMSE"].mean(),
        "EV": fr_v3["EV"].mean()
    }
    print(f"[OK] Loaded v3 fold report: MAE={local_metrics['v3']['MAE']:.5f}, RMSE={local_metrics['v3']['RMSE']:.5f}, EV={local_metrics['v3']['EV']:.5f}")
except:
    print("[MISS] v3 fold report not found")

# v10 fold report might not exist, check
try:
    fr_v10 = pd.read_csv("submissions/fold_report_v10.csv")
    local_metrics["v10"] = {
        "MAE": fr_v10["MAE"].mean(),
        "RMSE": fr_v10["RMSE"].mean(),
        "EV": fr_v10["EV"].mean()
    }
    print(f"[OK] Loaded v10 fold report")
except:
    print("[MISS] v10 fold report not found — will estimate from prediction stats")

print("\n--- Prediction Statistics ---")
pred_stats = {}
for name, info in submissions.items():
    try:
        df = pd.read_csv(info["file"])
        preds = df['flood_risk_score'].values
        stats = {
            "mean": float(np.mean(preds)),
            "std": float(np.std(preds)),
            "min": float(np.min(preds)),
            "max": float(np.max(preds)),
            "range": float(np.max(preds) - np.min(preds)),
            "q25": float(np.percentile(preds, 25)),
            "q75": float(np.percentile(preds, 75)),
            "iqr": float(np.percentile(preds, 75) - np.percentile(preds, 25)),
            "skew": float(pd.Series(preds).skew()),
            "kurtosis": float(pd.Series(preds).kurtosis()),
        }
        pred_stats[name] = stats
        pred_stats[name]["preds"] = preds
        print(f"  {name:>12}: mean={stats['mean']:.4f}  std={stats['std']:.4f}  "
              f"range=[{stats['min']:.4f}, {stats['max']:.4f}]  LB={info['lb']}")
    except Exception as e:
        print(f"  {name}: FAILED - {e}")

# =============================================================
# 2. PAIRWISE ANALYSIS — what changes between submissions?
# =============================================================
print("\n--- Pairwise Prediction Differences (MAD) ---")
names = list(pred_stats.keys())
for i, n1 in enumerate(names):
    for n2 in names[i+1:]:
        if n1 in pred_stats and n2 in pred_stats:
            mad = np.abs(pred_stats[n1]["preds"] - pred_stats[n2]["preds"]).mean()
            corr = np.corrcoef(pred_stats[n1]["preds"], pred_stats[n2]["preds"])[0,1]
            lb_diff = submissions[n2]["lb"] - submissions[n1]["lb"]
            print(f"  {n1:>12} vs {n2:<12}: MAD={mad:.5f}  corr={corr:.5f}  LB_diff={lb_diff:+.5f}")

# =============================================================
# 3. KEY INSIGHT: v10 vs v10_probe
# =============================================================
print("\n" + "=" * 70)
print("  KEY COMPARISON: v10 vs v10_probe (same model, different variance)")
print("=" * 70)

if "v10" in pred_stats and "v10_probe" in pred_stats:
    p10 = pred_stats["v10"]["preds"]
    p10k = pred_stats["v10_probe"]["preds"]
    
    # Check the stretching factor
    mean_10 = p10.mean()
    stretch_factors = (p10k - mean_10) / (p10 - mean_10 + 1e-10)
    print(f"  v10 mean:       {mean_10:.5f}")
    print(f"  v10 std:        {p10.std():.5f}")
    print(f"  v10_probe std:  {p10k.std():.5f}")
    print(f"  Stretch ratio:  {p10k.std() / p10.std():.3f}x")
    print(f"  v10 LB:         {submissions['v10']['lb']}")
    print(f"  v10_probe LB:   {submissions['v10_probe']['lb']}")
    print(f"  LB difference:  {submissions['v10_probe']['lb'] - submissions['v10']['lb']:+.5f}")
    print(f"  => Stretching variance by {p10k.std()/p10.std():.1f}x INCREASED error by {(submissions['v10_probe']['lb'] - submissions['v10']['lb'])/submissions['v10']['lb']*100:.1f}%")

# =============================================================
# 4. METRIC FORMULA FITTING
# =============================================================
print("\n" + "=" * 70)
print("  METRIC FORMULA FITTING")
print("=" * 70)

# We'll use prediction statistics as features since we can't compute true test metrics.
# For submissions with local metrics, we use those.
# Key insight: since all submissions predict the SAME test set, we can compute
# relative metrics between them.

# Build a reference: use v13 predictions as a proxy "anchor"
# Then compute relative stats of each submission vs v13

# But actually, let's try a different approach:
# Since the metric involves (predictions, true_labels), and we don't have true_labels,
# let's try to INFER what the metric formula is by using LOCAL metrics as proxies.

# For the v10_probe, we need to estimate local metrics.
# v10_probe was created by: pred_probe = mean + k*(pred_v10 - mean) where k=3.5
# If we know v10's local RMSE, then probe's RMSE ≈ k * RMSE (approximately)
# And EV is independent of scale (theoretically stays the same for linear transform)

# Actually EV DOES change with linear transforms when the transform is around
# a different center than the true mean. Let me think...

# EV = 1 - Var(y - ŷ) / Var(y)
# If ŷ_probe = mean_pred + k*(ŷ - mean_pred)
# Then y - ŷ_probe = y - mean_pred - k*(ŷ - mean_pred)
#                   = (y - ŷ) + (1-k)*(ŷ - mean_pred) + (ŷ - mean_pred) - (y - mean_pred) ... 
# This gets complex. Let's just compute numerically if we have v10's OOF.

# APPROACH: Use the 3 submissions with known local metrics (v11, v13, v17)
# plus the v10/v10_probe pair (which gives us a scale constraint).

# Let's try multiple formula forms and see which fits best.

# We have 3 clean data points: (v11, v13, v17) with known local metrics
# We need to find: LB = f(MAE_local, RMSE_local, EV_local) + noise

print("\nUsing v11, v13, v17 (known local metrics) to fit candidate formulas:\n")

# Data points
data_points = [
    ("v11", 0.17984, 0.23539, 0.02737, 0.38637),
    ("v13", 0.17937, 0.23500, 0.03060, 0.38476),
    ("v17", 0.17882, 0.23465, 0.03390, 0.38506),
]

names_fit = [d[0] for d in data_points]
mae_arr = np.array([d[1] for d in data_points])
rmse_arr = np.array([d[2] for d in data_points])
ev_arr = np.array([d[3] for d in data_points])
lb_arr = np.array([d[4] for d in data_points])

# Also add v3 if we have local metrics
if "v3" in local_metrics:
    data_points_ext = data_points + [("v3", local_metrics["v3"]["MAE"], local_metrics["v3"]["RMSE"], local_metrics["v3"]["EV"], 0.38559)]
    names_ext = [d[0] for d in data_points_ext]
    mae_ext = np.array([d[1] for d in data_points_ext])
    rmse_ext = np.array([d[2] for d in data_points_ext])
    ev_ext = np.array([d[3] for d in data_points_ext])
    lb_ext = np.array([d[4] for d in data_points_ext])
else:
    data_points_ext = data_points
    names_ext = names_fit
    mae_ext = mae_arr
    rmse_ext = rmse_arr
    ev_ext = ev_arr
    lb_ext = lb_arr

# ====== FORMULA CANDIDATES ======

formulas = {}

# F1: LB = RMSE * (1 + k*(1-EV))
def f1(params, mae, rmse, ev):
    k = params[0]
    return rmse * (1 + k * (1 - ev))
formulas["RMSE * (1 + k*(1-EV))"] = (f1, [0.65], 1)

# F2: LB = MAE * (1 + k*(1-EV))
def f2(params, mae, rmse, ev):
    k = params[0]
    return mae * (1 + k * (1 - ev))
formulas["MAE * (1 + k*(1-EV))"] = (f2, [1.15], 1)

# F3: LB = (a*MAE + b*RMSE) * (1 + k*(1-EV))
def f3(params, mae, rmse, ev):
    a, b, k = params
    base = a * mae + b * rmse
    return base * (1 + k * (1 - ev))
formulas["(a*MAE + b*RMSE) * (1+k*(1-EV))"] = (f3, [0.5, 0.5, 0.7], 3)

# F4: LB = sqrt(MAE * RMSE) * (1 + k*(1-EV))
def f4(params, mae, rmse, ev):
    k = params[0]
    return np.sqrt(mae * rmse) * (1 + k * (1 - ev))
formulas["sqrt(MAE*RMSE) * (1+k*(1-EV))"] = (f4, [0.85], 1)

# F5: LB = RMSE * (2 - EV)^k
def f5(params, mae, rmse, ev):
    k = params[0]
    return rmse * (2 - ev) ** k
formulas["RMSE * (2-EV)^k"] = (f5, [0.3], 1)

# F6: LB = (a*MAE + (1-a)*RMSE) * (2-EV)^k
def f6(params, mae, rmse, ev):
    a, k = params
    base = a * mae + (1 - a) * rmse
    return base * (2 - ev) ** k
formulas["(a*MAE+(1-a)*RMSE) * (2-EV)^k"] = (f6, [0.5, 0.3], 2)

# F7: LB = base_error + penalty, additive
def f7(params, mae, rmse, ev):
    a, b, c = params
    return a * mae + b * rmse + c * (1 - ev)
formulas["a*MAE + b*RMSE + c*(1-EV)"] = (f7, [0.5, 0.5, 0.1], 3)

# F8: LB = RMSE^a * (1/EV)^b
def f8(params, mae, rmse, ev):
    a, b = params
    return (rmse ** a) * ((1.0 / (ev + 0.001)) ** b)
formulas["RMSE^a * (1/EV)^b"] = (f8, [1.0, 0.01], 2)

# F9: LB = (MAE + RMSE)/2 * (2-EV)
def f9(params, mae, rmse, ev):
    return (mae + rmse) / 2 * (2 - ev)
formulas["(MAE+RMSE)/2 * (2-EV)"] = (f9, [], 0)

# F10: LB = a*MAE + b*RMSE + c*MAE*(1-EV) + d*RMSE*(1-EV)
def f10(params, mae, rmse, ev):
    a, b, c, d = params
    return a*mae + b*rmse + c*mae*(1-ev) + d*rmse*(1-ev)
formulas["a*MAE + b*RMSE + c*MAE*(1-EV) + d*RMSE*(1-EV)"] = (f10, [0.25, 0.25, 0.25, 0.25], 4)

# F11: LB = RMSE + k * RMSE * (1-EV)
def f11(params, mae, rmse, ev):
    k = params[0]
    return rmse + k * rmse * (1 - ev)
formulas["RMSE + k*RMSE*(1-EV)"] = (f11, [0.65], 1)

# F12: weighted harmonic/geometric mean approach
def f12(params, mae, rmse, ev):
    w, k = params
    base = (mae**w * rmse**(1-w))
    return base * (1 + k*(1-ev))
formulas["MAE^w * RMSE^(1-w) * (1+k*(1-EV))"] = (f12, [0.3, 0.8], 2)

print(f"Testing {len(formulas)} candidate formulas...\n")

# Fit each formula using the extended dataset
results = []
for name, (func, x0, n_params) in formulas.items():
    if n_params == 0:
        # No parameters to fit
        predicted = func([], mae_ext, rmse_ext, ev_ext)
        residuals = lb_ext - predicted
        mse = np.mean(residuals**2)
        max_err = np.max(np.abs(residuals))
        results.append((name, mse, max_err, [], predicted, residuals))
    else:
        def objective(params, func=func, mae=mae_ext, rmse=rmse_ext, ev=ev_ext, lb=lb_ext):
            pred = func(params, mae, rmse, ev)
            return np.sum((pred - lb)**2)
        
        # Try multiple random starts
        best_res = None
        best_mse = np.inf
        for _ in range(100):
            try:
                x0_rand = np.array(x0) * (0.5 + np.random.random(len(x0)))
                res = minimize(objective, x0_rand, method='Nelder-Mead', options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-12})
                if res.fun < best_mse:
                    best_mse = res.fun
                    best_res = res
            except:
                pass
        
        if best_res is not None:
            predicted = func(best_res.x, mae_ext, rmse_ext, ev_ext)
            residuals = lb_ext - predicted
            mse = np.mean(residuals**2)
            max_err = np.max(np.abs(residuals))
            results.append((name, mse, max_err, best_res.x, predicted, residuals))

# Sort by MSE
results.sort(key=lambda x: x[1])

print(f"{'Rank':<5} {'Formula':<50} {'MSE':>12} {'MaxErr':>10} {'Params'}")
print("-" * 100)
for rank, (name, mse, max_err, params, predicted, residuals) in enumerate(results, 1):
    param_str = ", ".join([f"{p:.6f}" for p in params]) if len(params) > 0 else "none"
    print(f"{rank:<5} {name:<50} {mse:>12.2e} {max_err:>10.5f} [{param_str}]")

# =============================================================
# 5. DETAILED ANALYSIS OF TOP FORMULAS
# =============================================================
print("\n" + "=" * 70)
print("  TOP 3 FORMULA DETAILED ANALYSIS")
print("=" * 70)

for rank, (name, mse, max_err, params, predicted, residuals) in enumerate(results[:3], 1):
    print(f"\n--- #{rank}: {name} ---")
    if len(params) > 0:
        print(f"  Parameters: {params}")
    print(f"  MSE: {mse:.2e}, Max Error: {max_err:.5f}")
    print(f"  {'Version':<12} {'LB Actual':>10} {'LB Predicted':>13} {'Residual':>10}")
    for i, n in enumerate(names_ext):
        print(f"  {n:<12} {lb_ext[i]:>10.5f} {predicted[i]:>13.5f} {residuals[i]:>+10.5f}")

# =============================================================
# 6. SENSITIVITY ANALYSIS — what matters most?
# =============================================================
print("\n" + "=" * 70)
print("  SENSITIVITY ANALYSIS (using best formula)")
print("=" * 70)

best_name, best_mse, best_max_err, best_params, best_pred, best_res = results[0]
best_func = formulas[best_name][0]

# Current best local metrics (v17)
base_mae, base_rmse, base_ev = 0.17882, 0.23465, 0.03390

print(f"\nBaseline (v17 local): MAE={base_mae}, RMSE={base_rmse}, EV={base_ev}")
print(f"Formula: {best_name}")
print(f"Params: {best_params}")

baseline_lb = best_func(best_params, base_mae, base_rmse, base_ev)
print(f"Predicted LB: {baseline_lb:.5f}")

print(f"\nSensitivity to each component:")
print(f"{'Change':<35} {'New LB':>10} {'Delta':>10}")

# MAE sensitivity
for delta in [-0.005, -0.001, +0.001, +0.005]:
    new_lb = best_func(best_params, base_mae + delta, base_rmse, base_ev)
    print(f"  MAE {delta:+.3f} → {base_mae+delta:.5f}       {new_lb:>10.5f} {new_lb-baseline_lb:>+10.5f}")

print()
# RMSE sensitivity
for delta in [-0.005, -0.001, +0.001, +0.005]:
    new_lb = best_func(best_params, base_mae, base_rmse + delta, base_ev)
    print(f"  RMSE {delta:+.3f} → {base_rmse+delta:.5f}      {new_lb:>10.5f} {new_lb-baseline_lb:>+10.5f}")

print()
# EV sensitivity
for delta in [-0.02, -0.01, +0.01, +0.02, +0.05, +0.10]:
    new_lb = best_func(best_params, base_mae, base_rmse, base_ev + delta)
    print(f"  EV {delta:+.03f} → {base_ev+delta:.5f}         {new_lb:>10.5f} {new_lb-baseline_lb:>+10.5f}")

# =============================================================
# 7. WHAT WOULD IT TAKE TO REACH RANK 1?
# =============================================================
print("\n" + "=" * 70)
print("  WHAT WOULD IT TAKE TO REACH RANK 1 (LB=0.38215)?")
print("=" * 70)

target_lb = 0.38215
print(f"\nTarget: {target_lb}")
print(f"Current best LB (v13): {0.38476}")
print(f"Gap: {0.38476 - target_lb:.5f}")

# If only EV changes
from scipy.optimize import brentq
try:
    def ev_objective(ev):
        return best_func(best_params, base_mae, base_rmse, ev) - target_lb
    
    needed_ev = brentq(ev_objective, 0.0, 1.0)
    print(f"\nIf only EV improves (MAE/RMSE fixed at v17): need EV = {needed_ev:.5f} (currently {base_ev:.5f}, need +{needed_ev-base_ev:.5f})")
except:
    print(f"\nCould not solve for EV-only path")

# If only RMSE changes
try:
    def rmse_objective(rmse):
        return best_func(best_params, base_mae, rmse, base_ev) - target_lb
    
    needed_rmse = brentq(rmse_objective, 0.01, 0.5)
    print(f"If only RMSE improves (MAE/EV fixed at v17):  need RMSE = {needed_rmse:.5f} (currently {base_rmse:.5f}, need {needed_rmse-base_rmse:+.5f})")
except:
    print(f"Could not solve for RMSE-only path")

# If only MAE changes
try:
    def mae_objective(mae):
        return best_func(best_params, mae, base_rmse, base_ev) - target_lb
    
    needed_mae = brentq(mae_objective, 0.01, 0.5)
    print(f"If only MAE improves (RMSE/EV fixed at v17):  need MAE = {needed_mae:.5f} (currently {base_mae:.5f}, need {needed_mae-base_mae:+.5f})")
except:
    print(f"Could not solve for MAE-only path")

print("\n" + "=" * 70)
print("  DONE")
print("=" * 70)
