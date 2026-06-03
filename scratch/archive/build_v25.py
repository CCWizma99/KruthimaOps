with open("scripts/train_v25.py", "r") as f:
    lines = f.readlines()

out = []
skip = False

for i, line in enumerate(lines):
    # Header update
    if "ML Opsidian: Genesis v24 - Median Align, Multi-Seed & Interaction Synergy" in line:
        line = line.replace("v24 - Median Align, Multi-Seed & Interaction Synergy", "v25 - Pure Statistical Foundation")
    if "ML OPSIDIAN v24 - TARGETED OVERFIT ROLLBACK" in line:
        line = line.replace("v24 - TARGETED OVERFIT ROLLBACK", "v25 - PURE STATISTICAL FOUNDATION")
    if "GLOBAL MULTI-SEED RESULTS (v24 - Targeted 7-Point Upgrade)" in line:
        line = line.replace("v24 - Targeted 7-Point Upgrade", "v25 - Pure Statistical Foundation")
        
    # Filenames update
    if "v24.csv" in line or "v24.npy" in line:
        line = line.replace("v24.csv", "v25.csv").replace("v24.npy", "v25.npy")
        
    # 1. Remove Precision Fingerprint block
    if "print(\"\\n[FEAT] Extracting precision fingerprint...\")" in line:
        skip = True
    if skip and "print(\"\\n[SEMI-SUPERVISED] Pseudo-Labeling from v20...\")" in line:
        skip = False
        
    # 2. Remove Pseudo-Labeling block
    if "print(\"\\n[SEMI-SUPERVISED] Pseudo-Labeling from v20...\")" in line:
        skip = True
    if skip and "print(\"\\n[IMPUTE] Geospatial Hot-Deck Imputation...\")" in line:
        skip = False
        
    if skip:
        continue
        
    # 3. Remove `is_pseudo` logic throughout
    if "'is_pseudo'" in line or "is_pseudo" in line:
        continue

    # Remove lat_decimal_len and lon_decimal_len from IGNORE_COLS
    if "lat_decimal_len" in line and "lon_decimal_len" in line:
        line = line.replace(", \"lat_decimal_len\", \"lon_decimal_len\"", "")

    # 4. Remove max_ctr_complexity=2 from CatBoost parameters (actually v24 didn't even have it! Let's just be sure)
    if "max_ctr_complexity=2" in line:
        line = line.replace(", max_ctr_complexity=2", "")
        
    # 5. Remove positive=True from Ridge
    if "Ridge(alpha=1.0, fit_intercept=True, positive=True)" in line:
        line = line.replace(", positive=True", "")
        
    # 6. Change variable references if they depended on real_mask
    if "real_mask" in line:
        # Since we removed is_pseudo, there's no real_mask. We can replace real_mask with nothing, but it's an array index.
        # Wait, if we remove real_mask entirely, we just replace `[real_mask]` with ``
        line = line.replace("[real_mask]", "")
        
    if "y_arr = y[real_mask].values" in line:
        line = "y_arr = y.values\n"
        
    if "real_tr_rows = tr_rows[tr_rows['is_pseudo'] == 0]" in line:
        line = "real_tr_rows = tr_rows\n"
        
    out.append(line)

with open("scripts/train_v25.py", "w") as f:
    f.writelines(out)
