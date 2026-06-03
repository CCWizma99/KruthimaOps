# Dataset Exploration & Taxonomy Report

> [!NOTE]
> This report contains a deep look at the tables, columns, distributions, and semantic meanings of features in the **ML Opsidian: Genesis** dataset.

## 1. Table Inventory

There are three primary data files in the `data/` directory:
- **[train.csv](file:///c:/KruthimaOps/data/train.csv)**: 20,886 rows, 47 columns. Contains the target variable `flood_risk_score`.
- **[test.csv](file:///c:/KruthimaOps/data/test.csv)**: 5,300 rows, 46 columns. Contains the same features as train, but lacks `flood_risk_score`.
- **[sample_submission.csv](file:///c:/KruthimaOps/data/sample_submission.csv)**: Format guide mapping each `record_id` to a target prediction.

### The Synthetic Smokescreen
- **Real vs Synthetic Split in Train:**
  - **Real Rows:** Only 802 rows (`is_synthetic` is `NaN`).
  - **Synthetic Rows:** 20,084 rows (`is_synthetic` is `True`).
- **Test Set:**
  - **100% Synthetic Rows:** All 5,300 rows are `is_synthetic = True`.

---
## 2. Detailed Column Analysis by Category

### Metadata & Administrative

| Column Name | Dtype | Nulls (Train %) | Nulls (Test %) | Uniq Vals | Samples / Stats |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `record_id` | str | 0 (0.0%) | 0 (0.0%) | 19700 | Samples: 'F100000', 'F100001', 'F100002' |
| `is_synthetic` | object | 802 (3.8%) | 0 (0.0%) | 1 | Samples: 'True', 'True', 'True' |
| `generation_date` | str | 0 (0.0%) | 0 (0.0%) | 730 | Samples: '2025-06-02', '2024-11-14', '2024-03-01' |

### Geographic & Location

| Column Name | Dtype | Nulls (Train %) | Nulls (Test %) | Uniq Vals | Samples / Stats |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `place_name` | str | 0 (0.0%) | 0 (0.0%) | 792 | Samples: 'Kudagama', 'Kiriwatta', 'Uduthota East' |
| `district` | str | 816 (3.9%) | 0 (0.0%) | 25 | Samples: 'Kegalle', 'Matara', 'Trincomalee' |
| `latitude` | float64 | 845 (4.0%) | 0 (0.0%) | 18862 | Range: [5.90, 9.95]<br>Mean: 7.92 +/- 1.17 |
| `longitude` | float64 | 840 (4.0%) | 0 (0.0%) | 18823 | Range: [79.65, 81.90]<br>Mean: 80.77 +/- 0.65 |

### Physical & Environmental

| Column Name | Dtype | Nulls (Train %) | Nulls (Test %) | Uniq Vals | Samples / Stats |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `elevation_m` | float64 | 791 (3.8%) | 0 (0.0%) | 1503 | Range: [-79.56, 2148.00]<br>Mean: 186.09 +/- 321.66 |
| `distance_to_river_m` | float64 | 1549 (7.4%) | 0 (0.0%) | 14823 | Range: [-485.40, 16802.00]<br>Mean: 2004.57 +/- 2021.92 |
| `landcover` | str | 809 (3.9%) | 0 (0.0%) | 7 | Samples: 'Wetland', 'Agriculture', 'Forest' |
| `soil_type` | str | 813 (3.9%) | 0 (0.0%) | 5 | Samples: 'Loamy', 'Clay', 'Silty' |
| `rainfall_7d_mm` | float64 | 820 (3.9%) | 0 (0.0%) | 2731 | Range: [-49.20, 914.82]<br>Mean: 84.76 +/- 79.51 |
| `monthly_rainfall_mm` | float64 | 811 (3.9%) | 0 (0.0%) | 5327 | Range: [-79.17, 2032.27]<br>Mean: 223.79 +/- 179.48 |
| `drainage_index` | float64 | 1315 (6.3%) | 0 (0.0%) | 1056 | Range: [0.00, 1.78]<br>Mean: 0.51 +/- 0.23 |
| `ndvi` | float64 | 1261 (6.0%) | 0 (0.0%) | 1532 | Range: [-0.79, 1.60]<br>Mean: 0.17 +/- 0.28 |
| `ndwi` | float64 | 1341 (6.4%) | 0 (0.0%) | 1549 | Range: [-1.00, 1.60]<br>Mean: 0.04 +/- 0.28 |
| `water_presence_flag` | str | 801 (3.8%) | 0 (0.0%) | 2 | Samples: 'Likely', 'Unlikely', 'Unlikely' |
| `historical_flood_count` | float64 | 804 (3.8%) | 0 (0.0%) | 6 | Range: [0.00, 5.00]<br>Mean: 0.20 +/- 0.51 |

### Human & Infrastructure

| Column Name | Dtype | Nulls (Train %) | Nulls (Test %) | Uniq Vals | Samples / Stats |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `water_supply` | str | 788 (3.8%) | 0 (0.0%) | 5 | Samples: 'Municipal', 'Municipal', 'Well' |
| `electricity` | str | 1604 (7.7%) | 0 (0.0%) | 3 | Samples: 'Mixed', 'Mixed', 'Mixed' |
| `road_quality` | str | 993 (4.8%) | 0 (0.0%) | 4 | Samples: 'Poor (unpaved)', 'Fair', 'Poor (unpaved)' |
| `population_density_per_km2` | float64 | 824 (3.9%) | 0 (0.0%) | 2453 | Range: [10.00, 6642.30]<br>Mean: 658.78 +/- 647.97 |
| `built_up_percent` | float64 | 801 (3.8%) | 0 (0.0%) | 921 | Range: [1.00, 178.26]<br>Mean: 25.67 +/- 19.94 |
| `urban_rural` | str | 799 (3.8%) | 0 (0.0%) | 2 | Samples: 'Rural', 'Rural', 'Rural' |
| `infrastructure_score` | float64 | 971 (4.6%) | 0 (0.0%) | 91 | Range: [5.00, 95.00]<br>Mean: 46.30 +/- 19.95 |
| `nearest_hospital_km` | float64 | 1023 (4.9%) | 0 (0.0%) | 3345 | Range: [0.00, 139.62]<br>Mean: 8.77 +/- 10.99 |
| `nearest_evac_km` | float64 | 804 (3.8%) | 0 (0.0%) | 2880 | Range: [0.00, 121.14]<br>Mean: 6.71 +/- 8.43 |

### Downstream Indicators (The 'Trap' columns)

| Column Name | Dtype | Nulls (Train %) | Nulls (Test %) | Uniq Vals | Samples / Stats |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `flood_occurrence_current_event` | str | 0 (0.0%) | 0 (0.0%) | 2 | Samples: 'Yes', 'Yes', 'No' |
| `inundation_area_sqm` | int64 | 0 (0.0%) | 0 (0.0%) | 10650 | Range: [396.00, 104489.00]<br>Mean: 7227.16 +/- 5645.27 |
| `is_good_to_live` | str | 0 (0.0%) | 0 (0.0%) | 2 | Samples: 'No', 'No', 'No' |
| `reason_not_good_to_live` | str | 804 (3.8%) | 217 (4.1%) | 8 | Samples: 'Other', 'High flood risk', 'Other' |

### Pre-engineered Transforms (Inside CSV)

| Column Name | Dtype | Nulls (Train %) | Nulls (Test %) | Uniq Vals | Samples / Stats |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `distance_to_river_m_log1p` | float64 | 1549 (7.4%) | 0 (0.0%) | 14823 | Range: [0.69, 9.76]<br>Mean: 7.54 +/- 0.76 |
| `population_density_per_km2_log1p` | float64 | 824 (3.9%) | 0 (0.0%) | 2453 | Range: [2.40, 8.80]<br>Mean: 5.99 +/- 1.24 |
| `rainfall_7d_mm_log1p` | float64 | 820 (3.9%) | 0 (0.0%) | 2731 | Range: [0.69, 6.87]<br>Mean: 4.79 +/- 0.49 |
| `monthly_rainfall_mm_log1p` | float64 | 811 (3.9%) | 0 (0.0%) | 5327 | Range: [0.69, 7.66]<br>Mean: 5.57 +/- 0.56 |
| `nearest_hospital_km_log1p` | float64 | 1023 (4.9%) | 0 (0.0%) | 3345 | Range: [0.00, 4.95]<br>Mean: 1.86 +/- 0.91 |
| `nearest_evac_km_log1p` | float64 | 804 (3.8%) | 0 (0.0%) | 2880 | Range: [0.00, 4.81]<br>Mean: 1.65 +/- 0.86 |
| `elevation_m_yeojohnson` | float64 | 791 (3.8%) | 0 (0.0%) | 1503 | Range: [-355.46, 147.48]<br>Mean: 25.96 +/- 30.58 |
| `drainage_index_yeojohnson` | float64 | 1315 (6.3%) | 0 (0.0%) | 1056 | Range: [0.00, 1.28]<br>Mean: 0.44 +/- 0.18 |
| `ndvi_qmap` | float64 | 1261 (6.0%) | 0 (0.0%) | 1532 | Range: [-5.20, 5.20]<br>Mean: 0.00 +/- 1.01 |
| `ndwi_qmap` | float64 | 1341 (6.4%) | 0 (0.0%) | 1545 | Range: [-5.20, 5.20]<br>Mean: 0.01 +/- 1.01 |
| `built_up_percent_qmap` | float64 | 801 (3.8%) | 0 (0.0%) | 915 | Range: [-5.20, 5.20]<br>Mean: -0.43 +/- 1.95 |
| `seasonal_index` | float64 | 0 (0.0%) | 0 (0.0%) | 19700 | Range: [-1.29, 1.34]<br>Mean: 0.00 +/- 0.72 |
| `terrain_roughness_index` | float64 | 0 (0.0%) | 0 (0.0%) | 19700 | Range: [0.18, 22.06]<br>Mean: 0.80 +/- 0.33 |
| `socioeconomic_status_index` | float64 | 0 (0.0%) | 0 (0.0%) | 19700 | Range: [0.19, 0.85]<br>Mean: 0.50 +/- 0.12 |
| `extreme_weather_index` | float64 | 0 (0.0%) | 0 (0.0%) | 19431 | Range: [0.00, 1.00]<br>Mean: 0.68 +/- 0.16 |

### Target Variable

| Column Name | Dtype | Nulls (Train %) | Nulls (Test %) | Uniq Vals | Samples / Stats |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `flood_risk_score` | float64 | 0 (0.0%) | N/A | 7397 | Range: [0.00, 1.00]<br>Mean: 0.48 +/- 0.24 |

---
## 3. Structural Anomalies & 'Traps' to Watch Out For

### Trap 1: The 'Downstream' Trap (Low Predictive Power of Physical Features)
> [!IMPORTANT]
> Standard physical features (like elevation, rainfall, lat/lon) are almost entirely uncorrelated with the target variable `flood_risk_score`. 
> The correlation table shows:
> - `distance_to_river_m_log1p`: correlation = +0.081 (Higher distance from river = slightly higher risk?)
> - `rainfall_7d_mm`: correlation = -0.052 (Higher rainfall = slightly lower risk?)
> - `extreme_weather_index`: correlation = -0.042
> 
> Instead, the model's true predictive power is concentrated in the **Downstream Indicators**:
> 1. `flood_occurrence_current_event`
> 2. `inundation_area_sqm`
> 3. `is_good_to_live`
> 4. `reason_not_good_to_live`

### Trap 2: Missing Values Contrast (Real vs. Synthetic)
> [!WARNING]
> - In **Real Rows** in the training set (802 rows), almost all environmental and human features have **~50% missing values**.
> - In **Synthetic Rows** in the training set (20,084 rows), the missing rate is much lower (**2% to 5%**).
> - In the **Test Set** (5,300 rows), there are **zero missing values** in environmental/human features (except `reason_not_good_to_live` which has 4.1% missing values).
> 
> This disparity can cause tree models to rely heavily on missingness indicators during training, which will not generalize to the test set because the test set has no missing values.

### Trap 3: Feature-Duplicate Contradiction
There are 2,372 rows where the exact same environmental and geographic features map to completely different `flood_risk_score` values. This indicates a high level of label noise in the dataset, capping the local Explained Variance (EV) at ~3.2%.