import os
import re

scripts_dir = "scripts"
files = sorted(os.listdir(scripts_dir))

train_files = []
for f in files:
    if f.endswith(".py") and ("train" in f.lower() or f == "Initial_trainer.py"):
        train_files.append(f)

# Sort files numerically
def get_num(filename):
    if filename == "Initial_trainer.py":
        return 1
    nums = re.findall(r'\d+', filename)
    if not nums:
        return 999
    # check if there's float like 3_5 -> 3.5
    if "3_5" in filename:
        return 3.5
    return float(nums[0])

train_files = sorted(train_files, key=get_num)

out_lines = []
for f in train_files:
    path = os.path.join(scripts_dir, f)
    out_lines.append(f"\n==================== {f} ====================\n")
    with open(path, "r", encoding="utf-8", errors="ignore") as file:
        lines = file.readlines()
    count = 0
    in_docstring = False
    for line in lines:
        if count > 45:
            break
        # Print lines that are comments, docstrings, or contain metadata info
        striped = line.strip()
        if striped.startswith('"""') or striped.startswith("'''"):
            in_docstring = not in_docstring
            out_lines.append(line)
            count += 1
            continue
        if in_docstring or striped.startswith("#") or "v" in line.lower() or "model" in line.lower() or "ensemble" in line.lower():
            out_lines.append(line)
        count += 1

with open("scratch/all_headers.txt", "w", encoding="utf-8") as f_out:
    f_out.writelines(out_lines)
print("Successfully wrote headers to scratch/all_headers.txt")
