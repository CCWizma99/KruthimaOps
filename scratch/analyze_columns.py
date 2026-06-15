import pandas as pd
df = pd.read_csv("c:/KruthimaOps/data/train.csv")
print("Columns in train.csv:")
print(list(df.columns))
print("\nFirst row sample:")
print(df.iloc[0].to_dict())
