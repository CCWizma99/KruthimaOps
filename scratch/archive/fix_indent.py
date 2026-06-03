with open("scripts/train_v24_kaggle.py", "r") as f:
    lines = f.readlines()

out = []
for line in lines:
    if line.startswith("    # sub_v19 is already in memory") or \
       line.startswith("    test_pseudo = test_df.merge") or \
       line.startswith("    # Filter highly confident predictions") or \
       line.startswith("    mask = (test_pseudo") or \
       line.startswith("    pseudo_rows = test_pseudo") or \
       line.startswith("    pseudo_rows['is_pseudo']") or \
       line.startswith("    train_df['is_pseudo']") or \
       line.startswith("    test_df['is_pseudo']") or \
       line.startswith("    print(f'   Added {len") or \
       line.startswith("    train_df = pd.concat([train_df") or \
       line.startswith("    # sub_v20 is already in memory") or \
       line.startswith("    test_pseudo = test_df.merge"):
        out.append(line[4:])
    else:
        out.append(line)

with open("scripts/train_v24_kaggle.py", "w") as f:
    f.writelines(out)
