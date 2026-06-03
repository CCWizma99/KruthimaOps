import os

# Base paths
v19_path = "scripts/train_v19_goldilocks.py"
v20_path = "scripts/train_v20_advanced.py"
v24_path = "scripts/train_v24.py"

with open(v19_path, "r") as f: v19_lines = f.readlines()
with open(v20_path, "r") as f: v20_lines = f.readlines()
with open(v24_path, "r") as f: v24_lines = f.readlines()

def clean_phase(lines, phase_num, next_phase_num=None):
    clean = []
    skip = False
    for line in lines:
        # Strip imports (we handle this globally)
        if line.startswith("import ") or line.startswith("from ") or "warnings.filterwarnings" in line:
            if phase_num != 1:  # Keep imports in Phase 1
                continue
                
        # Path replacements
        line = line.replace('"data/train.csv"', 'os.path.join(DATA_DIR, "train.csv")')
        line = line.replace('"data/test.csv"', 'os.path.join(DATA_DIR, "test.csv")')
        
        # Stop saving files at the end of intermediate phases
        if next_phase_num is not None:
            if "fold_report.to_csv" in line or "submission.to_csv" in line or "np.save(" in line:
                line = "# " + line
                
        # Handle Pseudo-Labeling blocks dynamically
        if 'print("\\n[SEMI-SUPERVISED] Pseudo-Labeling from v19...")' in line:
            clean.append(line)
            clean.append("    # sub_v19 is already in memory from Phase 1\n")
            clean.append("    test_pseudo = test_df.merge(sub_v19, on='record_id', how='left')\n")
            clean.append("    \n")
            clean.append("    # Filter highly confident predictions (around median)\n")
            clean.append("    mask = (test_pseudo['flood_risk_score'] >= 0.46) & (test_pseudo['flood_risk_score'] <= 0.49)\n")
            clean.append("    pseudo_rows = test_pseudo[mask].copy()\n")
            clean.append("    pseudo_rows['is_pseudo'] = 1\n")
            clean.append("    train_df['is_pseudo'] = 0\n")
            clean.append("    test_df['is_pseudo'] = 0\n")
            clean.append("    \n")
            clean.append("    print(f'   Added {len(pseudo_rows)} pseudo-labeled rows to training.')\n")
            clean.append("    train_df = pd.concat([train_df, pseudo_rows], ignore_index=True)\n")
            clean.append("    train_df['is_pseudo'] = 0\n")
            clean.append("    test_df['is_pseudo'] = 0\n")
            skip = True
            continue
            
        if 'print("\\n[SEMI-SUPERVISED] Pseudo-Labeling from v20...")' in line:
            clean.append(line)
            clean.append("    # sub_v20 is already in memory from Phase 2\n")
            clean.append("    test_pseudo = test_df.merge(sub_v20, on='record_id', how='left')\n")
            clean.append("    \n")
            clean.append("    # Filter highly confident predictions\n")
            clean.append("    mask = (test_pseudo['flood_risk_score'] <= 0.35) | (test_pseudo['flood_risk_score'] >= 0.60)\n")
            clean.append("    pseudo_rows = test_pseudo[mask].copy()\n")
            clean.append("    pseudo_rows['is_pseudo'] = 1\n")
            clean.append("    train_df['is_pseudo'] = 0\n")
            clean.append("    test_df['is_pseudo'] = 0\n")
            clean.append("    \n")
            clean.append("    print(f'   Added {len(pseudo_rows)} pseudo-labeled rows to training.')\n")
            clean.append("    train_df = pd.concat([train_df, pseudo_rows], ignore_index=True)\n")
            clean.append("    train_df['is_pseudo'] = 0\n")
            clean.append("    test_df['is_pseudo'] = 0\n")
            skip = True
            continue
            
        if skip and ('print("   [WARNING] submission_v19.csv not found' in line or 'print("   [WARNING] submission_v20.csv not found' in line):
            skip = False
            continue
            
        if skip and ('train_df[\'is_pseudo\']' in line or 'test_df[\'is_pseudo\']' in line):
            continue # already handled above
            
        if not skip:
            if phase_num == 3:
                line = line.replace('"submissions/fold_report_v24.csv"', '"fold_report_v24.csv"')
                line = line.replace('"submissions/submission_v24.csv"', '"submission_v24.csv"')
                line = line.replace('"submissions/oof_v24.npy"', '"oof_v24.npy"')
            clean.append(line)
            
        # Hook at end of phase to store predictions in memory
        if next_phase_num is not None:
            if "submission = pd.DataFrame" in line:
                pass # let it run to populate `submission`
                
    
    out_str = "".join(clean)
    
    if next_phase_num == 2:
        out_str += f"\nsub_v19 = submission.copy()\n"
        out_str += "del train_df, test_df, fold_report, submission  # Clear memory\n"
        out_str += "print('=' * 70)\nprint('  STARTING PHASE 2: ML OPSIDIAN v20')\nprint('=' * 70)\n\n"
    elif next_phase_num == 3:
        out_str += f"\nsub_v20 = submission.copy()\n"
        out_str += "del train_df, test_df, fold_report, submission  # Clear memory\n"
        out_str += "print('=' * 70)\nprint('  STARTING PHASE 3: ML OPSIDIAN v24')\nprint('=' * 70)\n\n"
        
    return out_str

header = """
# =================================================================
# MONOLITHIC KAGGLE PIPELINE (v19 -> v20 -> v24)
# Zero external files required. End-to-end pseudo-label generation.
# =================================================================
import os
DATA_DIR = "/kaggle/input/competitions/ml-opsidian-genesis-initial-round-26"
if not os.path.exists(DATA_DIR):
    DATA_DIR = "data" # Fallback local
"""

p1 = clean_phase(v19_lines, 1, next_phase_num=2)
p2 = clean_phase(v20_lines, 2, next_phase_num=3)
p3 = clean_phase(v24_lines, 3, next_phase_num=None)

with open("scripts/train_v24_kaggle.py", "w") as f:
    f.write(header)
    f.write(p1)
    f.write(p2)
    f.write(p3)

print("Successfully compiled Monolithic Kaggle Script.")
