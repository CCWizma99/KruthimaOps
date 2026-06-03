import re

with open('c:/KruthimaOps/scripts/train_v55_kaggle.py', 'r', encoding='utf-8') as f:
    code = f.read()

# 1. Update headers and versions
code = code.replace('v55', 'v56')
code = code.replace('v52', 'v55')  # Upgrades from v55 instead of v52

# 2. Update iterations and estimators to 800
code = code.replace('iterations=5000', 'iterations=800')
code = code.replace('iterations=4000', 'iterations=800')
code = code.replace('iterations=3000', 'iterations=800')
code = code.replace('n_estimators=4000', 'n_estimators=800')
code = code.replace('n_estimators=5000', 'n_estimators=800')

# 3. Update learning rate from 0.03 to 0.05
code = code.replace('learning_rate=0.03', 'learning_rate=0.05')

# 4. Update early_stopping_rounds to 100 uniformly
code = code.replace('early_stopping_rounds=150', 'early_stopping_rounds=100')

with open('c:/KruthimaOps/scripts/train_v56_kaggle.py', 'w', encoding='utf-8') as f:
    f.write(code)

print('Successfully generated train_v56_kaggle.py')
