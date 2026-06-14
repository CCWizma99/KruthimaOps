import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error


# --------------------------------------------------
# Paths
# --------------------------------------------------
ROOT = Path(".")
OUTPUT_DIR = ROOT / "outputs_blend"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_SUBMISSION = ROOT / "data" / "sample_submission.csv"

RUNS = [
    {
        "name": "v2",
        "oof_path": ROOT / "outputs_catboost_full_v2" / "full" / "catboost" / "oof_predictions.csv",
        "sub_path": ROOT / "outputs_catboost_full_v2" / "full" / "catboost" / "submission.csv",
    },
    {
        "name": "v3",
        "oof_path": ROOT / "outputs_catboost_full_v3" / "full" / "catboost" / "oof_predictions.csv",
        "sub_path": ROOT / "outputs_catboost_full_v3" / "full" / "catboost" / "submission.csv",
    },
    {
        "name": "rankgauss",
        "oof_path": ROOT / "outputs_catboost_full_rankgauss" / "full" / "catboost" / "oof_predictions.csv",
        "sub_path": ROOT / "outputs_catboost_full_rankgauss" / "full" / "catboost" / "submission.csv",
    },
]


# --------------------------------------------------
# Helper functions
# --------------------------------------------------
def validate_submission(submission: pd.DataFrame, sample: pd.DataFrame) -> None:
    required_cols = ["record_id", "flood_risk_score"]

    if list(submission.columns) != required_cols:
        raise ValueError("Submission columns must be exactly: record_id, flood_risk_score")

    if len(submission) != len(sample):
        raise ValueError("Submission row count does not match sample_submission.csv")

    if not submission["record_id"].equals(sample["record_id"]):
        raise ValueError("record_id order does not match sample_submission.csv")

    if submission.isna().any().any():
        raise ValueError("Submission contains missing values")

    if not submission["flood_risk_score"].between(0, 1).all():
        raise ValueError("flood_risk_score values must be between 0 and 1")

    if submission["record_id"].duplicated().any():
        raise ValueError("Submission contains duplicate record_id values")


def grid_weights(step: float = 0.05):
    values = np.arange(0, 1 + step, step)

    for w_v2 in values:
        for w_v3 in values:
            w_rankgauss = 1.0 - w_v2 - w_v3

            if w_rankgauss < -1e-9:
                continue

            if w_rankgauss > 1 + 1e-9:
                continue

            yield round(w_v2, 2), round(w_v3, 2), round(w_rankgauss, 2)


# --------------------------------------------------
# Load sample submission
# --------------------------------------------------
sample = pd.read_csv(SAMPLE_SUBMISSION)

if "record_id" not in sample.columns or "flood_risk_score" not in sample.columns:
    raise ValueError("sample_submission.csv must contain record_id and flood_risk_score")


# --------------------------------------------------
# Load OOF and submission files
# --------------------------------------------------
oof_frames = []
sub_frames = []

for i, run in enumerate(RUNS):
    oof_df = pd.read_csv(run["oof_path"])
    sub_df = pd.read_csv(run["sub_path"])

    if not {"record_id", "oof_pred", "flood_risk_score"}.issubset(oof_df.columns):
        raise ValueError(
            f"{run['oof_path']} must contain record_id, oof_pred, and flood_risk_score"
        )

    if not {"record_id", "flood_risk_score"}.issubset(sub_df.columns):
        raise ValueError(
            f"{run['sub_path']} must contain record_id and flood_risk_score"
        )

    # Keep true target only from first OOF file to avoid flood_risk_score_x/y merge issue
    if i == 0:
        oof_frames.append(
            oof_df[["record_id", "oof_pred", "flood_risk_score"]]
            .rename(columns={"oof_pred": f"pred_{run['name']}"})
        )
    else:
        oof_frames.append(
            oof_df[["record_id", "oof_pred"]]
            .rename(columns={"oof_pred": f"pred_{run['name']}"})
        )

    sub_frames.append(
        sub_df[["record_id", "flood_risk_score"]]
        .rename(columns={"flood_risk_score": f"pred_{run['name']}"})
    )


# --------------------------------------------------
# Merge OOF predictions
# --------------------------------------------------
merged_oof = oof_frames[0]

for frame in oof_frames[1:]:
    merged_oof = merged_oof.merge(frame, on="record_id", how="inner")

if len(merged_oof) == 0:
    raise ValueError("OOF merge produced empty dataframe; record_id alignment failed")


# --------------------------------------------------
# Merge test predictions
# --------------------------------------------------
merged_sub = sub_frames[0]

for frame in sub_frames[1:]:
    merged_sub = merged_sub.merge(frame, on="record_id", how="inner")


# Validate merged test prediction order using one prediction column
validate_submission(
    merged_sub[["record_id", "pred_v2"]].rename(
        columns={"pred_v2": "flood_risk_score"}
    ),
    sample,
)


# --------------------------------------------------
# Prepare arrays
# --------------------------------------------------
y_true = merged_oof["flood_risk_score"].to_numpy(dtype=float)

preds = {
    "v2": merged_oof["pred_v2"].to_numpy(dtype=float),
    "v3": merged_oof["pred_v3"].to_numpy(dtype=float),
    "rankgauss": merged_oof["pred_rankgauss"].to_numpy(dtype=float),
}

test_preds = {
    "v2": merged_sub["pred_v2"].to_numpy(dtype=float),
    "v3": merged_sub["pred_v3"].to_numpy(dtype=float),
    "rankgauss": merged_sub["pred_rankgauss"].to_numpy(dtype=float),
}


# --------------------------------------------------
# Search best blend weights
# --------------------------------------------------
results = []

for w_v2, w_v3, w_rankgauss in grid_weights(0.05):
    blended_oof = (
        w_v2 * preds["v2"]
        + w_v3 * preds["v3"]
        + w_rankgauss * preds["rankgauss"]
    )

    blended_oof = np.clip(blended_oof, 0.0, 1.0)

    mae = mean_absolute_error(y_true, blended_oof)
    rmse = np.sqrt(mean_squared_error(y_true, blended_oof))
    ev = 1.0 - np.var(y_true - blended_oof) / np.var(y_true)

    # Smaller is better
    balanced_score = (rmse + (1.0 - ev)) / 2.0

    results.append(
        {
            "w_v2": w_v2,
            "w_v3": w_v3,
            "w_rankgauss": w_rankgauss,
            "mae": mae,
            "rmse": rmse,
            "ev": ev,
            "balanced_score": balanced_score,
        }
    )


results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    ["mae", "rmse", "ev"],
    ascending=[True, True, False],
).reset_index(drop=True)

results_df.head(20).to_csv(OUTPUT_DIR / "blend_results.csv", index=False)

best_mae = results_df.iloc[0]

best_balanced = results_df.sort_values(
    ["balanced_score", "mae", "rmse"],
    ascending=[True, True, True],
).iloc[0]


# --------------------------------------------------
# Create best MAE submission
# --------------------------------------------------
best_mae_preds = (
    best_mae["w_v2"] * test_preds["v2"]
    + best_mae["w_v3"] * test_preds["v3"]
    + best_mae["w_rankgauss"] * test_preds["rankgauss"]
)

best_mae_preds = np.clip(best_mae_preds, 0.0, 1.0)

best_mae_submission = pd.DataFrame(
    {
        "record_id": sample["record_id"],
        "flood_risk_score": best_mae_preds,
    }
)


# --------------------------------------------------
# Create balanced submission
# --------------------------------------------------
balanced_preds = (
    best_balanced["w_v2"] * test_preds["v2"]
    + best_balanced["w_v3"] * test_preds["v3"]
    + best_balanced["w_rankgauss"] * test_preds["rankgauss"]
)

balanced_preds = np.clip(balanced_preds, 0.0, 1.0)

balanced_submission = pd.DataFrame(
    {
        "record_id": sample["record_id"],
        "flood_risk_score": balanced_preds,
    }
)


# --------------------------------------------------
# Validate and save submissions
# --------------------------------------------------
validate_submission(best_mae_submission, sample)
validate_submission(balanced_submission, sample)

best_mae_submission.to_csv(OUTPUT_DIR / "submission_blend_best_mae.csv", index=False)
balanced_submission.to_csv(OUTPUT_DIR / "submission_blend_balanced.csv", index=False)

# Also save final submission in project root
best_mae_submission.to_csv(ROOT / "final_submission.csv", index=False)


# --------------------------------------------------
# Print summary
# --------------------------------------------------
print("Top 20 blends written to:", OUTPUT_DIR / "blend_results.csv")

print("\nBest MAE blend:")
print(
    f"  weights -> v2={best_mae['w_v2']:.2f}, "
    f"v3={best_mae['w_v3']:.2f}, "
    f"rankgauss={best_mae['w_rankgauss']:.2f}"
)
print(
    f"  metrics -> MAE={best_mae['mae']:.6f}, "
    f"RMSE={best_mae['rmse']:.6f}, "
    f"EV={best_mae['ev']:.6f}"
)

print("\nBest balanced blend:")
print(
    f"  weights -> v2={best_balanced['w_v2']:.2f}, "
    f"v3={best_balanced['w_v3']:.2f}, "
    f"rankgauss={best_balanced['w_rankgauss']:.2f}"
)
print(
    f"  metrics -> MAE={best_balanced['mae']:.6f}, "
    f"RMSE={best_balanced['rmse']:.6f}, "
    f"EV={best_balanced['ev']:.6f}"
)

print("\nFinal submission saved to:", ROOT / "final_submission.csv")
print("Best MAE submission saved to:", OUTPUT_DIR / "submission_blend_best_mae.csv")
print("Balanced submission saved to:", OUTPUT_DIR / "submission_blend_balanced.csv")