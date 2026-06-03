import numpy as np
import pandas as pd
import shutil

# Optimal parameters obtained from scripts/post_optimize.py:
a = 1.02699
b = 1.01412
c = -0.00818

# Load and optimize OOF predictions
oof_path = "submissions/oof_v43.npy"
raw_oof = np.load(oof_path)
opt_oof = a * np.power(np.clip(raw_oof, 1e-6, None), b) + c
opt_oof = np.clip(opt_oof, 0.0, 1.0)
np.save("submissions/oof_v43_optimized.npy", opt_oof)
print("Saved submissions/oof_v43_optimized.npy")

# Copy the optimized submission file to the root directory
shutil.copy("submissions/submission_v43_optimized.csv", "submission_v43_optimized.csv")
print("Copied optimized submission to submission_v43_optimized.csv (root)")
