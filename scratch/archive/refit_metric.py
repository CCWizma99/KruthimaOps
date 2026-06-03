"""Re-fit the competition metric formula with 5 data points (2 DOF)."""
import numpy as np
from scipy.optimize import minimize

# All 5 known LB submissions
data = [
    # (MAE,     RMSE,    EV,      actual_LB)
    (0.17962, 0.23520, 0.02889, 0.38559),  # v3
    (0.17984, 0.23539, 0.02737, 0.38637),  # v11
    (0.17937, 0.23500, 0.03060, 0.38476),  # v13
    (0.17882, 0.23465, 0.03390, 0.38506),  # v17
    (0.17891, 0.23461, 0.03379, 0.38401),  # v19
]

mae_arr  = np.array([d[0] for d in data])
rmse_arr = np.array([d[1] for d in data])
ev_arr   = np.array([d[2] for d in data])
lb_arr   = np.array([d[3] for d in data])

# === Formula: LB = a*MAE + b*RMSE + c*(1-EV) ===
A = np.column_stack([mae_arr, rmse_arr, 1.0 - ev_arr])
params_ls, residuals, rank, sv = np.linalg.lstsq(A, lb_arr, rcond=None)
a, b, c = params_ls

pred = A @ params_ls
errors = lb_arr - pred

print("=" * 60)
print("  RE-FITTED FORMULA (5 data points, 2 DOF)")
print("=" * 60)
print(f"\n  LB = {a:.6f} * MAE + {b:.6f} * RMSE + {c:.6f} * (1-EV)")
print(f"\n  Validation:")
names = ["v3", "v11", "v13", "v17", "v19"]
for i, name in enumerate(names):
    print(f"    {name:<5}: predicted={pred[i]:.5f}  actual={lb_arr[i]:.5f}  error={errors[i]:+.5f}")

print(f"\n  MSE: {np.mean(errors**2):.2e}")
print(f"  Max Error: {np.max(np.abs(errors)):.5f}")
print(f"  RMS Error: {np.sqrt(np.mean(errors**2)):.5f}")

# Sensitivity
print(f"\n  Marginal values:")
print(f"    dLB/dMAE  = {a:.2f}  (MAE drop of 0.001 = {-a*0.001:.5f} LB improvement)")
print(f"    dLB/dRMSE = {b:.2f}  (RMSE drop of 0.001 = {-b*0.001:.5f} LB improvement)")
print(f"    dLB/dEV   = {-c:.2f}  (EV rise of 0.01 = {c*0.01:.5f} LB improvement)")
print(f"    MAE/RMSE weight ratio: {abs(a/b):.2f}x")

# === Also try other formula forms ===
print("\n" + "=" * 60)
print("  ALTERNATIVE FORMULAS")
print("=" * 60)

# F2: LB = (a*MAE + b*RMSE) * (1 + k*(1-EV))
def f2(params):
    a2, b2, k = params
    pred2 = (a2 * mae_arr + b2 * rmse_arr) * (1 + k * (1 - ev_arr))
    return np.sum((pred2 - lb_arr)**2)

best = None
best_val = np.inf
for _ in range(10):
    x0 = np.random.randn(3) * 0.5
    res = minimize(f2, x0, method='Nelder-Mead', options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8})
    if res.fun < best_val:
        best_val = res.fun
        best = res

if best:
    a2, b2, k = best.x
    pred2 = (a2 * mae_arr + b2 * rmse_arr) * (1 + k * (1 - ev_arr))
    errors2 = lb_arr - pred2
    print(f"\n  Multiplicative: ({a2:.6f}*MAE + {b2:.6f}*RMSE) * (1 + {k:.6f}*(1-EV))")
    for i, name in enumerate(names):
        print(f"    {name:<5}: predicted={pred2[i]:.5f}  actual={lb_arr[i]:.5f}  error={errors2[i]:+.5f}")
    print(f"  MSE: {np.mean(errors2**2):.2e}, Max Error: {np.max(np.abs(errors2)):.5f}")

# F3: RMSE * (1 + k*(1-EV))
def f3(params):
    k = params[0]
    pred3 = rmse_arr * (1 + k * (1 - ev_arr))
    return np.sum((pred3 - lb_arr)**2)

best3 = minimize(f3, [0.65], method='Nelder-Mead', options={'maxiter': 50000})
k3 = best3.x[0]
pred3 = rmse_arr * (1 + k3 * (1 - ev_arr))
errors3 = lb_arr - pred3
print(f"\n  Simple: RMSE * (1 + {k3:.6f}*(1-EV))")
for i, name in enumerate(names):
    print(f"    {name:<5}: predicted={pred3[i]:.5f}  actual={lb_arr[i]:.5f}  error={errors3[i]:+.5f}")
print(f"  MSE: {np.mean(errors3**2):.2e}, Max Error: {np.max(np.abs(errors3)):.5f}")

print(f"\n{'=' * 60}")
print(f"  COPY-PASTE FOR evaluate.py:")
print(f"{'=' * 60}")
print(f"  return {a:.6f} * mae + {b:.6f} * rmse + {c:.6f} * (1.0 - ev)")
