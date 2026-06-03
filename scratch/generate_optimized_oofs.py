import numpy as np

# Define calibration mappings
calibrations = {
    "v37": (0.96265, 1.46268, 0.15000),
    "v38": (1.00106, 1.51612, 0.15000),
    "v42": (0.98864, 1.49940, 0.15000),
    "v43": (1.02699, 1.01412, -0.00818)
}

for ver, (a, b, c) in calibrations.items():
    raw_oof = np.load(f"submissions/oof_{ver}.npy")
    opt_oof = a * np.power(np.clip(raw_oof, 1e-6, None), b) + c
    opt_oof = np.clip(opt_oof, 0.0, 1.0)
    np.save(f"submissions/oof_{ver}_optimized.npy", opt_oof)
    print(f"Saved submissions/oof_{ver}_optimized.npy using a={a}, b={b}, c={c}")
