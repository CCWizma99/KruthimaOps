import os

with open("scripts/train_v20_advanced.py", "r") as f:
    v20_code = f.read()

with open("scripts/train_v24.py", "r") as f:
    v24_code = f.read()

# 1. Clean v20
# Remove imports
v20_lines = v20_code.split("\n")
v20_clean = []
for line in v20_lines:
    if line.startswith("import ") or line.startswith("from "):
        continue
    if "warnings.filterwarnings" in line:
        continue
    if "fold_report.to_csv" in line or "submission.to_csv" in line or "np.save(" in line:
        line = "# " + line
    if "submission = pd.DataFrame" in line:
        v20_clean.append("print('\\n[PHASE 1 COMPLETE] Extracting v20 pseudo labels...')")
    line = line.replace('"data/train.csv"', 'os.path.join(DATA_DIR, "train.csv")')
    line = line.replace('"data/test.csv"', 'os.path.join(DATA_DIR, "test.csv")')
    v20_clean.append(line)

v20_final = "\n".join(v20_clean)
v20_final += "\n\nsub_v20 = submission.copy()\n"
v20_final += "del train_df, test_df, fold_report, submission\n"
v20_final += "print('=' * 70)\nprint('  STARTING PHASE 2: ML OPSIDIAN v24')\nprint('=' * 70)\n\n"

# 2. Clean v24
v24_lines = v24_code.split("\n")
v24_clean = []
skip = False
for line in v24_lines:
    if line.startswith("import ") or line.startswith("from ") or "warnings.filterwarnings" in line:
        continue
    
    line = line.replace('"data/train.csv"', 'os.path.join(DATA_DIR, "train.csv")')
    line = line.replace('"data/test.csv"', 'os.path.join(DATA_DIR, "test.csv")')
    line = line.replace('"submissions/fold_report_v24.csv"', '"fold_report_v24.csv"')
    line = line.replace('"submissions/submission_v24.csv"', '"submission_v24.csv"')
    line = line.replace('"submissions/oof_v24.npy"', '"oof_v24.npy"')
    
    if 'print("\\n[SEMI-SUPERVISED] Pseudo-Labeling from v20...")' in line:
        v24_clean.append(line)
        v24_clean.append("    # sub_v20 is already in memory from Phase 1")
        v24_clean.append("    test_pseudo = test_df.merge(sub_v20, on='record_id', how='left')")
        v24_clean.append("    ")
        v24_clean.append("    # Filter highly confident predictions (around median)")
        v24_clean.append("    mask = (test_pseudo['flood_risk_score'] >= 0.46) & (test_pseudo['flood_risk_score'] <= 0.49)")
        v24_clean.append("    pseudo_rows = test_pseudo[mask].copy()")
        v24_clean.append("    pseudo_rows['is_pseudo'] = 1")
        v24_clean.append("    train_df['is_pseudo'] = 0")
        v24_clean.append("    test_df['is_pseudo'] = 0")
        v24_clean.append("    ")
        v24_clean.append("    print(f'   Added {len(pseudo_rows)} pseudo-labeled rows to training.')")
        v24_clean.append("    train_df = pd.concat([train_df, pseudo_rows], ignore_index=True)")
        skip = True
        continue
        
    if skip and 'print("   [WARNING] submission_v20.csv not found' in line:
        skip = False
        continue
        
    if not skip:
        v24_clean.append(line)

v24_final = "\n".join(v24_clean)

# 3. Compile
header = """
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, explained_variance_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
import xgboost as xgb
import catboost as cb
import warnings
import time
import os

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------
# 0. KAGGLE ENVIRONMENT SETUP
# -----------------------------------------------------------------
DATA_DIR = "/kaggle/input/competitions/ml-opsidian-genesis-initial-round-26"
if not os.path.exists(DATA_DIR):
    DATA_DIR = "data" # Fallback local
"""

with open("scripts/train_v24_kaggle.py", "w") as f:
    f.write(header)
    f.write(v20_final)
    f.write(v24_final)

print("Fixed merge.")
