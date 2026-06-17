import re

file_path = "c:/KruthimaOps/training/serialize_pipeline_v1k2.py"
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# Update OUTPUT_DIR
content = content.replace(
    'OUTPUT_DIR  = os.path.join("production", "models", "prod_v1000")',
    'OUTPUT_DIR  = os.path.join("production", "models", "prod_v1k.2")'
)

# Update metadata version
content = content.replace("'version':          'prod_v1000'", "'version':          'prod_v1k.2'")
content = content.replace("'base_pipeline':    'v1000'", "'base_pipeline':    'v1k.2'")

# Add features inside engineer_features
feature_patch = """
    df["rain_spike_ratio"]            = df["rainfall_7d_mm"] / (df["monthly_rainfall_mm"] + 1e-6)
    
    # --- v1k.2 Engine Features ---
    df['topographical_vulnerability'] = df['elevation_m'] / (df['distance_to_river_m'] + 1.0)
    
    pop_density = np.expm1(df["population_density_per_km2_log1p"])
    is_urban = (df['urban_rural'].astype(str).str.strip().str.lower() == 'urban').astype(int)
    df['urban_runoff_intensity'] = df['rainfall_7d_mm'] * pop_density * is_urban
    
    soil_perm_map = {'Sandy': 3, 'Loamy': 2, 'Silty': 2, 'Clay': 1, 'Peaty': 1}
    df['soil_permeability_score'] = df['soil_type'].astype(str).map(soil_perm_map).fillna(2)
    
    df['infrastructure_deficit_v2'] = df['infrastructure_score'] * pop_density
    df['is_monsoon_peak'] = df['month'].isin([5, 6, 11, 12]).astype(int)
    # -----------------------------
"""
content = content.replace(
    '    df["rain_spike_ratio"]            = df["rainfall_7d_mm"] / (df["monthly_rainfall_mm"] + 1e-6)',
    feature_patch
)

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)

print("v1k.2 changes applied to script.")
