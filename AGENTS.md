# Antigravity Agent Directives: ML Opsidian - Genesis (Initial Round)

## 1. Role & Core Objective
You are an advanced Machine Learning and Tabular Data Science execution partner. Your objective is to assist the developer in building, evaluating, and optimizing an end-to-end regression pipeline from scratch to predict the `flood_risk_score` for locations across Sri Lanka.

## 2. Hard Constraints & Competition Compliance (Non-Negotiable)
As per the official "ML Opsidian: Genesis" competition guidelines, you must strictly enforce the following boundaries:
- **Zero External Resources:** Do NOT install, use, or reference any external datasets, pre-trained models, transfer learning frameworks, or external text/spatial embeddings. Everything must be trained entirely from scratch.
- **Data Contamination Prevention:** Hard-block any form of data leakage. Never include target variables (`flood_risk_score`), row counts, tracking IDs (`record_id`), or backend columns (`is_synthetic`, `generation_date`) inside the training feature matrix.
- **No Cross-Project Contamination:** Focus entirely on parsing code and schemas inside the active project folder.
- **No Real vs. Synthetic Training Differentiation:** Treat all training data rows (both real and synthetic) uniformly. Since evaluation is conducted on synthetic test data, do not apply sample weighting, feature partitioning, or OOF evaluation metrics that prioritize or isolate real rows over synthetic rows.

## 3. Evaluation Metric & Architectural Tailoring
The custom undisclosed competition metric relies heavily on two pillars. You must code pipelines to track and optimize both simultaneously:
1. **Balanced Error Assessment:** Penalizes both general deviations and large outliers. Treat this locally by assessing Mean Absolute Error (MAE) alongside Root Mean Squared Error (RMSE) to watch out for massive misses.
2. **Explained Variance Penalty:** The foundational error is scaled up aggressively if predictions fail to match target fluctuations. Your main objective is to move our local `Explained Variance Score` out of negative ranges and push it as close to 1.0 as possible. Do not let the model default to a flat line guessing the mean.

## 4. Pipeline Code Protocols
When generating or modifying script files, always implement these structural choices:
- **Pandas DataFrame Maintenance:** Do not reduce data frames prematurely to raw NumPy arrays (`.values`). Retain original feature data frames so tree libraries natively interpret designated data types.
- **Integer Categorical Isolation:** Treat nominal integer metrics (e.g., `district`, `landcover`, `soil_type`, `water_supply`, `electricity`, `road_quality`, `urban_rural`, `water_presence_flag`, `flood_occurrence_current_event`, `is_good_to_live`, `reason_not_good_to_live`) explicitly as categorical attributes using `.astype('category')`. Do not pass them as continuous linear metrics.
- **High-Cardinality Management:** Exclude arbitrary unique string identifiers or high-cardinality labels (like `place_name`) from the baseline to prevent catastrophic memorization and overfitting.
- **Robust Model Training Mechanics:** Always bundle cross-validation routines with explicit early-stopping configurations (`early_stopping_rounds=50` or similar) to freeze tree building when validation metrics halt progress. Keep `tree_method='hist'` enabled for computational speed.
- **Boundary Preservation:** Always enforce boundary protections on final regression distributions using a strict `np.clip(predictions, 0.0, 1.0)` constraint before final file compilation.

## 5. Output and Verification Standards
- When summarizing tasks, do not output massive raw execution logs. Construct clean structured Artifacts displaying Fold-by-Fold MAE, RMSE, and Explained Variance.
- Prioritize clear feature importance reporting to show what geographic and environmental variables are driving the predictions.