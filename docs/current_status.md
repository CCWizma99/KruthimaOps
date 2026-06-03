# Current Status Report: ML Opsidian Genesis (Initial Round)

## 1. The Scenario & Objective
- **Competition:** "ML Opsidian: Genesis" (Tabular Data Science Hackathon).
- **Objective:** Predict the continuous `flood_risk_score` (ranging from `0.0` to `1.0`) for locations across Sri Lanka.
- **Constraints:** Zero external datasets or pre-trained models allowed. Must build everything from scratch.
- **The Evaluation Metric (The Trap):** The competition uses a custom, undisclosed evaluation metric based on two pillars:
  1. **Balanced Error Assessment:** Strictly penalizes absolute deviations and large outliers (RMSE/MAE).
  2. **Explained Variance Penalty:** Aggressively scales up the foundational error if the prediction variance fails to match the wide fluctuations of the true target variance. (i.e., do not let the model default to a flat line guessing the mean).

## 2. Dataset Profile & Structural Mechanics
- **Shape:** Train = 20,886 rows | Test = 5,300 rows.
- **The Synthetic Smokescreen:** 
  - 20,084 training rows are flagged as `is_synthetic = True`.
  - Only 802 training rows are real (`is_synthetic = NaN`).
  - However, **100% of the 5,300 test rows** are flagged as `is_synthetic = True`, masking which rows the leaderboard actually scores.
- **Landmine 1: Feature-Duplicate Contradiction:** We ran diagnostics and found **2,372 rows** where the exact same geographic and environmental features produced wildly conflicting target scores. This mathematically forces models to predict the mean for those combinations.
- **Landmine 2: The Downstream Trap:** There are 4 "downstream" indicators (`flood_occurrence_current_event`, `inundation_area_sqm`, `is_good_to_live`, `reason_not_good_to_live`). We proved that the raw physical environment (Rainfall, Lat/Lon, Elevation) has absolutely zero predictive power (Negative 26% EV). The entirety of the predictable signal resides strictly within those 4 downstream categorical features.

## 3. What We Built & Tested
1. **The Core Pipeline (v3):** We built a highly robust Triple Tree Ensemble (XGBoost, LightGBM, CatBoost) featuring K-Fold Target Encoding (for spatial regularizers like District), polynomial interactions, and spatial aggregations.
2. **Deep Learning Hypothesis (v5):** We hypothesized the signal was a continuous mathematical equation trees couldn't map. We built a Deep Neural Network (MLP) with Lasso L1 pruning. **Result:** Failed. The MLP hit the exact same 3% EV ceiling as the trees.
3. **Ground Truth Isolation (v6 & v8):** We hypothesized the 20,084 synthetic rows were destroying the signal. We trained exclusively on the 802 Real Rows. **Result:** Failed. The models stopped at iteration 0, proving the real rows are just as noisy and unpredictable as the synthetic data.
4. **Ridge Regularization Check:** We ran a heavily regularized Ridge Regression (`alpha=100.0`) to see if the true generative formula was a simple flat linear line. **Result:** Failed. Ridge achieved 0.5% EV.

## 4. The Results
- **The Mathematical Noise Ceiling:** Regardless of the algorithm (Linear, Non-Linear, Tree, MLP), the absolute maximum Explained Variance (EV) achievable on this dataset is a hard **3% (0.03)**. 
- **The Safe Prediction Range:** Because the data is structurally 97% random noise, the models naturally protect their RMSE by defaulting to the mean. Our highly-optimized models output a tight, conservative prediction range of `[0.39, 0.58]`.
- **Local Error Metrics:** Our lowest, most stable out-of-fold RMSE is **`0.2350`**.
- **Public Leaderboard (20% Split):** We submitted our baseline `submission_v3.csv`. We immediately hit **Rank 3 with a score of 0.38559** (Rank 1 is 0.38215). 

## 5. Our Strategic Assumptions & Defensive Posture
*This section outlines our final decision to submit a low-variance model despite the competition's variance penalty.*

**Assumption A: Stretching Predictions is a Fatal Math Trap**
The competition rules try to goad competitors into artificially widening their predictions to match the target's `[0.0, 1.0]` fluctuations. We assumed this is a trap based on the variance-covariance expansion formula:
- Our current RMSE is $0.235$. The correlation ($\rho$) is $\approx 0.17$.
- If we stretch predictions to span the true target variance ($Var = 0.238$), our new MSE becomes: $MSE_{new} = Var(y) + Var(\tilde{y}) - 2 \cdot Cov(y, \tilde{y})$
- By calculating the math: $0.0566 + 0.0566 - 2(0.0096) = 0.094$. The new $RMSE = \sqrt{0.094} \approx 0.306$.
- **Why we assumed this:** Artificially stretching predictions skyrockets our foundational RMSE from `0.235` to `0.306`. We assume this massive explosion in absolute error will completely overwhelm any minor relief we get from dodging the variance scaling penalty.

**Assumption B: The High-Noise Leaderboard Shakeup**
We assumed the public leaderboard is a lottery designed to punish teams that overfit to noise. 
- **Why we assumed this:** Because the EV is capped at 3% and Rank 1 has a score of `0.382`, the variance penalty is actively punishing the entire leaderboard. We assume the teams slightly above us are deploying high-variance, over-tuned models to chase that public score. 
- **The Defensive Strategy:** When the competition ends and evaluates on the 80% Private Split, those overfitted, high-variance models will suffer catastrophic outlier misses. By locking in our highly-regularized, safe RMSE baseline (0.235), we are playing the ultimate defensive strategy. We assume our score will remain structurally stable during the mass private-leaderboard shakeup, naturally carrying us to the top.
