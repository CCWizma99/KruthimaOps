import pandas as pd
import json
import os
import time
import pdf_extractor

print("Loading dataset to find required dates...")
df = pd.read_csv('c:/KruthimaOps/data/train_v1002_desinventar.csv')
real_df = df[df['is_synthetic'].isna()].copy()
real_df['generation_date'] = pd.to_datetime(real_df['generation_date'])

# Filter out dates before our index (though we might have PDFs for them)
# Actually, let's just get all unique dates
unique_dates = real_df['generation_date'].dt.strftime('%Y-%m-%d').unique()
print(f"Found {len(unique_dates)} unique historical dates.")

print("Loading DMC PDF Index...")
try:
    with open('c:/KruthimaOps/data/dmc_pdf_index.json') as f:
        pdf_index = json.load(f)
except Exception as e:
    print("Could not load pdf_index. Run dmc_indexer.py first.")
    exit(1)

offline_data = {}
OUTPUT_FILE = 'c:/KruthimaOps/data/offline_river_gauges.json'

# Load existing progress if any
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, 'r') as f:
        offline_data = json.load(f)
    print(f"Resuming from {len(offline_data)} already compiled dates.")

total_to_process = len(unique_dates)
processed = 0
found = 0

print("\nStarting Bulk Offline Compilation...")
for d in unique_dates:
    if d in offline_data:
        processed += 1
        found += 1
        continue
        
    if d in pdf_index:
        pdf_url = pdf_index[d]
        pdf_path = pdf_extractor.download_pdf(pdf_url, d)
        
        if pdf_path:
            gauge_data = pdf_extractor.extract_gauge_data(pdf_path)
            if gauge_data:
                offline_data[d] = gauge_data
                found += 1
                
        # Be nice to the DMC server
        time.sleep(1)
    else:
        # If no exact date match, we could implement a fuzzy match (e.g. previous day),
        # but for now we just record it as empty
        offline_data[d] = {}
        
    processed += 1
    if processed % 10 == 0:
        print(f"Progress: {processed}/{total_to_process} (Found PDFs for {found})")
        # Save checkpoints
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(offline_data, f, indent=4)

# Final save
with open(OUTPUT_FILE, 'w') as f:
    json.dump(offline_data, f, indent=4)
    
print(f"\nBulk Compilation Complete. Compiled data for {found} out of {total_to_process} dates.")
print(f"Offline dataset saved to: {OUTPUT_FILE}")
