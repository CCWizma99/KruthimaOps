import pandas as pd

tr = pd.read_csv('data/train.csv').dropna(subset=['latitude', 'longitude'])
te = pd.read_csv('data/test.csv')

for res in [1.0, 0.5, 0.25, 0.125, 0.0625]:
    tr_grids = set(zip((tr['latitude'] // res).astype(int), (tr['longitude'] // res).astype(int)))
    te_grids = set(zip((te['latitude'] // res).astype(int), (te['longitude'] // res).astype(int)))
    overlap = len(tr_grids.intersection(te_grids))
    print(f"Res {res}: Train unique grids={len(tr_grids)}, Test unique grids={len(te_grids)}, Overlap={overlap}")
