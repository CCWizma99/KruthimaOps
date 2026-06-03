import pandas as pd
import zlib
import base64
import os

# 1. Compress the pristine v20 submission
df = pd.read_csv('submissions/submission_v20.csv')
payload = base64.b64encode(zlib.compress(df.to_csv(index=False).encode('utf-8'))).decode('ascii')

# 2. Read v24 source code
with open('scripts/train_v24.py', 'r') as f:
    v24_code = f.read()

v24_lines = v24_code.split('\n')
v24_clean = []
skip = False

for line in v24_lines:
    if line.startswith("import ") or line.startswith("from ") or "warnings.filterwarnings" in line:
        continue
    
    # Path mappings
    line = line.replace('"data/train.csv"', 'os.path.join(DATA_DIR, "train.csv")')
    line = line.replace('"data/test.csv"', 'os.path.join(DATA_DIR, "test.csv")')
    line = line.replace('"submissions/fold_report_v24.csv"', '"fold_report_v24.csv"')
    line = line.replace('"submissions/submission_v24.csv"', '"submission_v24.csv"')
    line = line.replace('"submissions/oof_v24.npy"', '"oof_v24.npy"')

    # Inject the embedded decoding logic right before pseudo-labeling
    if 'print("\\n[SEMI-SUPERVISED] Pseudo-Labeling from v20...")' in line:
        v24_clean.append(line)
        v24_clean.append("    print('   [INFO] Decoding embedded pristine v20 predictions...')")
        v24_clean.append("    import zlib, base64, io")
        v24_clean.append("    csv_data = zlib.decompress(base64.b64decode(V20_PAYLOAD)).decode('utf-8')")
        v24_clean.append("    sub_v20 = pd.read_csv(io.StringIO(csv_data))")
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


# 3. Construct Final Kaggle Script
header = f"""
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

# =================================================================
# EMBEDDED PRISTINE v20 PREDICTIONS (No external files needed)
# Contains the exact output of v20 (which used v19 pseudo-labels)
# =================================================================
V20_PAYLOAD = "{payload}"
"""

with open("scripts/train_v24_kaggle.py", "w") as f:
    f.write(header.strip())
    f.write("\n\n")
    f.write(v24_final.strip())

print("Successfully injected payload into scripts/train_v24_kaggle.py")
