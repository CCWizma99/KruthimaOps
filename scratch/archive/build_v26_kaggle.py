import sys

with open("scripts/train_v26.py", "r") as f:
    content = f.read()

# 1. Update DATA_DIR loading
import_os_str = """import time
import os"""

kaggle_data_str = """import time
import os

DATA_DIR = "/kaggle/input/competitions/ml-opsidian-genesis-initial-round-26"
if not os.path.exists(DATA_DIR):
    DATA_DIR = "data" # Fallback local
"""
content = content.replace(import_os_str, kaggle_data_str)

load_old = """print("\\n[LOAD] Loading data...")
train_df = pd.read_csv("data/train.csv")
test_df  = pd.read_csv("data/test.csv")"""

load_new = """print("\\n[LOAD] Loading data...")
train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test_df  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))"""
content = content.replace(load_old, load_new)

# 2. Update save paths (remove 'submissions/')
content = content.replace('"submissions/fold_report_v26.csv"', '"fold_report_v26.csv"')
content = content.replace('"submissions/submission_v26.csv"', '"submission_v26.csv"')
content = content.replace('"submissions/oof_v26.npy"', '"oof_v26.npy"')

with open("scripts/train_v26_kaggle.py", "w") as f:
    f.write(content)
