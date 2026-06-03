import numpy as np
import pandas as pd
from scipy.optimize import minimize
import os

# Ground truth
train_df = pd.read_csv("data/train.csv")
train_df = train_df.drop_duplicates()
y = train_df['flood_risk_score'].values

# Metrics for the 6 versions
data = {
    "v13":        {"mae": 0.179369, "rmse": 0.234996, "ev": 0.030601, "lb": 0.38476},
    "v17":        {"mae": 0.178821, "rmse": 0.234653, "ev": 0.033904, "lb": 0.38506},
    "v19":        {"mae": 0.178908, "rmse": 0.234610, "ev": 0.033785, "lb": 0.38401},
    "v20":        {"mae": 0.178645, "rmse": 0.234385, "ev": 0.035641, "lb": 0.38331},
    "v23":        {"mae": 0.178799, "rmse": 0.234493, "ev": 0.034752, "lb": 0.38411},
    "v28_kaggle": {"mae": 0.179288, "rmse": 0.234787, "ev": 0.032374, "lb": 0.38499},
    "v30":        {"mae": 0.178620, "rmse": 0.234360, "ev": 0.035870, "lb": 0.38293},
    "v33":        {"mae": 0.178630, "rmse": 0.234440, "ev": 0.035190, "lb": 0.38294}
}

# Also let's check if we can add v3 and v11 to the fit:
# v3:  MAE=0.179666, RMSE=0.235254, EV=0.028473, LB=0.385590 (from PathWeGo.txt)
# v11: MAE=0.179842, RMSE=0.235389, EV=0.027370, LB=0.386370 (from PathWeGo.txt)
data["v3"] = {"mae": 0.179666, "rmse": 0.235254, "ev": 0.028473, "lb": 0.385590}
data["v11"] = {"mae": 0.179842, "rmse": 0.235389, "ev": 0.027370, "lb": 0.386370}

# Let's define the models to fit:

# Model 1: Standard Multiplicative: (w0 * MAE + w1 * RMSE) * (1.0 + w2 * (1.0 - EV))
def model_1(w, mae, rmse, ev):
    return (w[0] * mae + w[1] * rmse) * (1.0 + w[2] * (1.0 - ev))

# Model 2: Multiplicative with intercept: (w0 * MAE + w1 * RMSE) * (1.0 + w2 * (1.0 - EV)) + w3
def model_2(w, mae, rmse, ev):
    return (w[0] * mae + w[1] * rmse) * (1.0 + w[2] * (1.0 - ev)) + w[3]

# Model 3: Multiplicative with offset in penalty: (w0 * MAE + w1 * RMSE) * (w2 + w3 * (1.0 - EV))
def model_3(w, mae, rmse, ev):
    return (w[0] * mae + w[1] * rmse) * (w[2] + w[3] * (1.0 - ev))

# Model 4: Pure MAE Multiplicative with intercept: MAE * (w0 + w1 * (1.0 - EV)) + w2
def model_4(w, mae, rmse, ev):
    return mae * (w[0] + w[1] * (1.0 - ev)) + w[2]

# Model 5: Pure RMSE Multiplicative with intercept: RMSE * (w0 + w1 * (1.0 - EV)) + w2
def model_5(w, mae, rmse, ev):
    return rmse * (w[0] + w[1] * (1.0 - ev)) + w[2]

# Model 6: Linear Model: w0 * MAE + w1 * RMSE + w2 * (1.0 - EV) + w3
def model_6(w, mae, rmse, ev):
    return w[0] * mae + w[1] * rmse + w[2] * (1.0 - ev) + w[3]

# Fit and print results for each model
models = [
    ("Model 1: (w0*MAE + w1*RMSE) * (1.0 + w2*(1-EV))", model_1, [1.0, 1.0, 1.0], [(0, None), (0, None), (0, None)]),
    ("Model 2: (w0*MAE + w1*RMSE) * (1.0 + w2*(1-EV)) + w3", model_2, [1.0, 1.0, 1.0, 0.0], [(0, None), (0, None), (0, None), (None, None)]),
    ("Model 3: (w0*MAE + w1*RMSE) * (w2 + w3*(1-EV))", model_3, [1.0, 1.0, 1.0, 1.0], [(0, None), (0, None), (0, None), (0, None)]),
    ("Model 4: MAE * (w0 + w1*(1-EV)) + w2", model_4, [1.0, 1.0, 0.0], [(0, None), (0, None), (None, None)]),
    ("Model 5: RMSE * (w0 + w1*(1-EV)) + w2", model_5, [1.0, 1.0, 0.0], [(0, None), (0, None), (None, None)]),
    ("Model 6: Linear Model: w0*MAE + w1*RMSE + w2*(1-EV) + w3", model_6, [1.0, 1.0, 1.0, 0.0], [(None, None), (None, None), (None, None), (None, None)])
]

for name, func, init, bounds in models:
    def loss(w):
        err = 0
        for ver, val in data.items():
            pred = func(w, val["mae"], val["rmse"], val["ev"])
            err += (pred - val["lb"]) ** 2
        return err / len(data)
    
    res = minimize(loss, init, bounds=bounds, method='L-BFGS-B')
    w_opt = res.x
    
    print("\n" + "="*60)
    print(name)
    print("="*60)
    max_err = 0
    for ver, val in sorted(data.items()):
        pred = func(w_opt, val["mae"], val["rmse"], val["ev"])
        err = pred - val["lb"]
        max_err = max(max_err, abs(err))
        print(f"  {ver:<10} -> Pred: {pred:.5f} | Act: {val['lb']:.5f} | Err: {err:+.5f}")
    print(f"  Optimal weights: {w_opt}")
    print(f"  Max Absolute Error: {max_err:.5f}")

# Let's perform a global OLS linear regression on:
# LB = w0 * MAE + w1 * RMSE + w2 * (1 - EV) + w3
print("\n" + "="*60)
print("OLS Linear Regression: w0*MAE + w1*RMSE + w2*(1-EV) + w3")
print("="*60)
X = []
Y = []
versions = []
for ver, val in sorted(data.items()):
    X.append([val["mae"], val["rmse"], 1.0 - val["ev"]])
    Y.append(val["lb"])
    versions.append(ver)
X = np.array(X)
Y = np.array(Y)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X, Y)
w = lr.coef_
intercept = lr.intercept_

print(f"Coefficients: MAE={w[0]:.6f}, RMSE={w[1]:.6f}, 1-EV={w[2]:.6f}")
print(f"Intercept: {intercept:.6f}")

max_err = 0
for i, ver in enumerate(versions):
    pred = lr.predict(X[i:i+1])[0]
    err = pred - Y[i]
    max_err = max(max_err, abs(err))
    print(f"  {ver:<10} -> Pred: {pred:.5f} | Act: {Y[i]:.5f} | Err: {err:+.5f}")
print(f"  Max Absolute Error: {max_err:.5f}")


# Multi-start optimization for Model 1 to ensure global convergence
print("\n" + "="*60)
print("Multi-Start Global Optimization for Model 1 (MSE)")
print("="*60)

best_loss = float('inf')
best_w = None

# We can define the objective function
def model_1_loss(w):
    err = 0
    for ver, val in data.items():
        pred = (w[0] * val["mae"] + w[1] * val["rmse"]) * (1.0 + w[2] * (1.0 - val["ev"]))
        err += (pred - val["lb"]) ** 2
    return err / len(data)

# Run 100 random restarts
np.random.seed(42)
for _ in range(100):
    init = np.random.uniform(0.1, 2.0, size=3)
    res = minimize(model_1_loss, init, bounds=[(0, None), (0, None), (0, None)], method='L-BFGS-B')
    if res.fun < best_loss:
        best_loss = res.fun
        best_w = res.x

print(f"Best Loss (MSE): {best_loss:.10f}")
print(f"Optimal weights: w0 (MAE) = {best_w[0]:.6f}, w1 (RMSE) = {best_w[1]:.6f}, w2 (1-EV Penalty) = {best_w[2]:.6f}")
print(f"Formula to copy:")
print(f"  LB = ({best_w[0]:.6f} * MAE + {best_w[1]:.6f} * RMSE) * (1.0 + {best_w[2]:.6f} * (1.0 - EV))")

max_err = 0
for ver, val in sorted(data.items()):
    pred = (best_w[0] * val["mae"] + best_w[1] * val["rmse"]) * (1.0 + best_w[2] * (1.0 - val["ev"]))
    err = pred - val["lb"]
    max_err = max(max_err, abs(err))
    print(f"  {ver:<10} -> Pred: {pred:.5f} | Act: {val['lb']:.5f} | Err: {err:+.5f}")
print(f"  Max Absolute Error: {max_err:.5f}")


# Minimax optimization for Model 1 (Minimize Max Absolute Error)
print("\n" + "="*60)
print("Minimax Optimization for Model 1 (Minimize Max Absolute Error)")
print("="*60)

# We can define the objective function for minimax
def model_1_minimax_loss(w):
    max_err = 0
    for ver, val in data.items():
        pred = (w[0] * val["mae"] + w[1] * val["rmse"]) * (1.0 + w[2] * (1.0 - val["ev"]))
        err = abs(pred - val["lb"])
        if err > max_err:
            max_err = err
    return max_err

# We can enforce that MAE and RMSE coefficients are both positive and reasonably balanced,
# say w0 >= 0.2 and w1 >= 0.2 to prevent either from collapsing to 0.
bounds = [(0.2, None), (0.2, None), (0.0, None)]

best_minimax_loss = float('inf')
best_minimax_w = None

for _ in range(100):
    init = np.random.uniform(0.2, 1.5, size=3)
    res = minimize(model_1_minimax_loss, init, bounds=bounds, method='L-BFGS-B')
    if res.fun < best_minimax_loss:
        best_minimax_loss = res.fun
        best_minimax_w = res.x

print(f"Best Minimax Loss (Max Abs Error): {best_minimax_loss:.6f}")
print(f"Optimal weights: w0 (MAE) = {best_minimax_w[0]:.6f}, w1 (RMSE) = {best_minimax_w[1]:.6f}, w2 (1-EV Penalty) = {best_minimax_w[2]:.6f}")
print(f"Formula to copy:")
print(f"  LB = ({best_minimax_w[0]:.6f} * MAE + {best_minimax_w[1]:.6f} * RMSE) * (1.0 + {best_minimax_w[2]:.6f} * (1.0 - EV))")

max_err = 0
for ver, val in sorted(data.items()):
    pred = (best_minimax_w[0] * val["mae"] + best_minimax_w[1] * val["rmse"]) * (1.0 + best_minimax_w[2] * (1.0 - val["ev"]))
    err = pred - val["lb"]
    max_err = max(max_err, abs(err))
    print(f"  {ver:<10} -> Pred: {pred:.5f} | Act: {val['lb']:.5f} | Err: {err:+.5f}")
print(f"  Max Absolute Error: {max_err:.5f}")


# Convex Combination fitting
print("\n" + "="*60)
print("Convex Combination Minimax Optimization: scale * (alpha * MAE + (1-alpha) * RMSE) * (1 + beta * (1-EV))")
print("="*60)

def convex_minimax_loss(w):
    # w = [scale, alpha, beta]
    scale, alpha, beta = w[0], w[1], w[2]
    max_err = 0
    for ver, val in data.items():
        foundational_err = alpha * val["mae"] + (1.0 - alpha) * val["rmse"]
        pred = scale * foundational_err * (1.0 + beta * (1.0 - val["ev"]))
        err = abs(pred - val["lb"])
        if err > max_err:
            max_err = err
    return max_err

best_conv_loss = float('inf')
best_conv_w = None

for _ in range(100):
    init = [np.random.uniform(0.5, 2.0), np.random.uniform(0.0, 1.0), np.random.uniform(0.1, 2.0)]
    res = minimize(convex_minimax_loss, init, bounds=[(0.1, None), (0.0, 1.0), (0.0, None)], method='L-BFGS-B')
    if res.fun < best_conv_loss:
        best_conv_loss = res.fun
        best_conv_w = res.x

scale_opt, alpha_opt, beta_opt = best_conv_w[0], best_conv_w[1], best_conv_w[2]
w0_opt = scale_opt * alpha_opt
w1_opt = scale_opt * (1.0 - alpha_opt)
w2_opt = beta_opt

print(f"Best Loss (Max Abs Error): {best_conv_loss:.6f}")
print(f"Optimal parameters: scale = {scale_opt:.6f}, alpha (MAE weight) = {alpha_opt:.6f}, beta (Penalty) = {beta_opt:.6f}")
print(f"Implied w0 (MAE coeff) = {w0_opt:.6f}, Implied w1 (RMSE coeff) = {w1_opt:.6f}")
print(f"Formula:")
print(f"  LB = ({w0_opt:.6f} * MAE + {w1_opt:.6f} * RMSE) * (1.0 + {w2_opt:.6f} * (1.0 - EV))")

max_err = 0
for ver, val in sorted(data.items()):
    pred = (w0_opt * val["mae"] + w1_opt * val["rmse"]) * (1.0 + w2_opt * (1.0 - val["ev"]))
    err = pred - val["lb"]
    max_err = max(max_err, abs(err))
    print(f"  {ver:<10} -> Pred: {pred:.5f} | Act: {val['lb']:.5f} | Err: {err:+.5f}")
print(f"  Max Absolute Error: {max_err:.5f}")





