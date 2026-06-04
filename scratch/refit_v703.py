import numpy as np
from scipy.optimize import minimize

data = [
    ("v3",          0.179666, 0.235254, 0.028473, 0.385590),
    ("v11",         0.179842, 0.235389, 0.027370, 0.386370),
    ("v13",         0.179369, 0.235000, 0.030601, 0.384760),
    ("v17",         0.178821, 0.234653, 0.033904, 0.385060),
    ("v19",         0.178908, 0.234610, 0.033785, 0.384010),
    ("v20",         0.178645, 0.234385, 0.035641, 0.383310),
    ("v23",         0.178799, 0.234493, 0.034752, 0.384110),
    ("v28",         0.179288, 0.234787, 0.032374, 0.384990),
    ("v30",         0.178620, 0.234360, 0.035870, 0.382930),
    ("v33",         0.178630, 0.234440, 0.035190, 0.382940),
    ("v37",         0.178530, 0.234390, 0.035660, 0.383350),
    ("v37opt",      0.178510, 0.234360, 0.035890, 0.383280),
    ("v38opt",      0.178530, 0.234380, 0.035710, 0.382980),
    ("v42opt",      0.178110, 0.234010, 0.038730, 0.382450),
    ("v44opt",      0.178060, 0.233990, 0.038920, 0.382780),
    ("v45opt",      0.178110, 0.234020, 0.038690, 0.382720),
    ("v54opt",      0.178190, 0.234200, 0.037510, 0.383370),
    ("v60opt",      0.178270, 0.234120, 0.037820, 0.382950),
    ("v63opt",      0.178220, 0.234080, 0.038220, 0.383090),
    ("v64opt",      0.178030, 0.233930, 0.039420, 0.382560),
    ("v67opt",      0.178030, 0.233960, 0.039190, 0.382160),
    ("v70opt",      0.178030, 0.233950, 0.039230, 0.382160),
    ("v77opt",      0.178060, 0.233960, 0.039140, 0.382530),
    ("v80opt",      0.178060, 0.233960, 0.039150, 0.382400),
    # v703_optimized - actual LB 0.382030 (NEW PEAK)
    ("v703opt",     0.178066, 0.233989, 0.038936, 0.382030),
]

maes  = np.array([d[1] for d in data])
rmses = np.array([d[2] for d in data])
evs   = np.array([d[3] for d in data])
lbs   = np.array([d[4] for d in data])

def sim(params, mae, rmse, ev):
    c1, c2, c3, c4 = params
    return (c1 * mae + c2 * rmse) * (1.0 + c3 * (1.0 - ev)) + c4

def loss(params):
    return float(np.sum((sim(params, maes, rmses, evs) - lbs)**2))

res = minimize(loss, [0.563017, 1.168101, 0.141013, -0.041721], method='L-BFGS-B')
c1, c2, c3, c4 = res.x
preds = sim(res.x, maes, rmses, evs)
rms = float(np.sqrt(np.mean((preds - lbs)**2)))
max_dev = float(np.max(np.abs(preds - lbs)))

print("=== REFITTED WITH v703 (25 data points) ===")
print(f"  c_mae={c1:.6f}, c_rmse={c2:.6f}, c_ev={c3:.6f}, c_int={c4:.6f}")
print(f"  RMS Dev: {rms:.5f}  |  Max Abs Dev: {max_dev:.5f}")
print()

print(f"  {'Version':<15} {'Actual':>8} {'Est':>8} {'Dev':>8}")
print(f"  {'-'*46}")
for i, row in enumerate(data):
    ver = row[0]
    marker = " <<< NEW" if ver == "v703opt" else ""
    print(f"  {ver:<15} {lbs[i]:>8.5f} {preds[i]:>8.5f} {preds[i]-lbs[i]:>+8.5f}{marker}")

print()
print("=== OLD vs NEW COEFFICIENTS ===")
print(f"  Old: c_mae=0.563017, c_rmse=1.168101, c_ev=0.141013, c_int=-0.041721")
print(f"  New: c_mae={c1:.6f}, c_rmse={c2:.6f}, c_ev={c3:.6f}, c_int={c4:.6f}")
