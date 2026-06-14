import os
import shutil
import glob

# Define kept versions (as prefixes or substrings)
KEPT_VERSIONS = {
    "v1000",
    "v58",
    "vb58",
    "v703_hub",
    "v703_7m",
    "v703_hub_oof_te",
    "v38",
    "v703"
}

# Directories
ROOT_DIR = "c:/KruthimaOps"
SUBMISSIONS_DIR = os.path.join(ROOT_DIR, "submissions")
SCRIPTS_DIR = os.path.join(ROOT_DIR, "scripts")

ARCHIVE_SUBMISSIONS = os.path.join(SUBMISSIONS_DIR, "archive")
ARCHIVE_SCRIPTS = os.path.join(SCRIPTS_DIR, "archive")

os.makedirs(ARCHIVE_SUBMISSIONS, exist_ok=True)
os.makedirs(ARCHIVE_SCRIPTS, exist_ok=True)

# Helper function to check if a file belongs to a kept version
def is_kept_version(filename):
    fn = filename.lower()
    for v in KEPT_VERSIONS:
        v_lower = v.lower()
        # Check if the version is in the filename as a distinct block
        # Matches "_v38.", "_v38_", "v38.csv", "v38.npy" etc.
        if f"_{v_lower}." in fn or f"_{v_lower}_" in fn or fn.startswith(f"submission_{v_lower}") or fn.startswith(f"oof_{v_lower}") or fn.startswith(f"fold_report_{v_lower}"):
            return True
    return False

print("--- Step 1: Archiving non-kept scripts ---")
# 1. Handle scripts/ folder
for filepath in glob.glob(os.path.join(SCRIPTS_DIR, "*.py")):
    basename = os.path.basename(filepath)
    if basename.startswith("train_"):
        # Determine if it is a kept script
        if not is_kept_version(basename):
            print(f"Archiving script: {basename}")
            shutil.move(filepath, os.path.join(ARCHIVE_SCRIPTS, basename))

print("\n--- Step 2: Archiving non-kept submission folder files ---")
# 2. Handle submissions/ folder
for filepath in glob.glob(os.path.join(SUBMISSIONS_DIR, "*")):
    if os.path.isdir(filepath):
        continue  # skip subdirectories (archive, probes, etc.)
    basename = os.path.basename(filepath)
    if not is_kept_version(basename):
        print(f"Archiving submission file: {basename}")
        shutil.move(filepath, os.path.join(ARCHIVE_SUBMISSIONS, basename))

print("\n--- Step 3: Handling root folder outputs ---")
# 3. Handle root folder outputs (CSV, NPY files and similar output patterns)
root_patterns = [
    "fold_report_*.csv",
    "oof_*.npy",
    "submission_*.csv",
    "adversarial_v2_importance.csv",
    "feature_importance_fast_fun.csv"
]

for pattern in root_patterns:
    for filepath in glob.glob(os.path.join(ROOT_DIR, pattern)):
        basename = os.path.basename(filepath)
        if is_kept_version(basename):
            # Duplicate of a kept file, we delete it from root since it is already in submissions/
            print(f"Deleting root duplicate of kept file: {basename}")
            os.remove(filepath)
        else:
            # Non-kept output in root, we move it to submissions/archive/
            print(f"Archiving root output file: {basename}")
            # Target path in archive
            target_path = os.path.join(ARCHIVE_SUBMISSIONS, basename)
            # If the file already exists in archive (from step 2), delete the root one, otherwise move it
            if os.path.exists(target_path):
                print(f"File already in archive submissions, removing root duplicate: {basename}")
                os.remove(filepath)
            else:
                shutil.move(filepath, target_path)

print("\nCleanup successfully complete!")
