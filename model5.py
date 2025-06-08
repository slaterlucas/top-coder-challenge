#!/usr/bin/env python3
"""
advanced_gam_rounded.py

Fits a piece-wise segmented “GAM-style” model to the reimbursement data,
then applies quarter-cent rounding (nearest $0.25) to the predictions,
as determined by the rounding analysis.
"""

import json
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import joblib

# 1) Define your knots
receipt_knots = [6, 451, 923, 1406, 1600, 1800, 1992]
# mpd
mpd_knots     = [1, 37, 67, 103, 181, 400, 600, 800, 1000]

spd_knots     = [2, 70, 135, 199, 345]
days_knots    = [2, 4, 6, 8, 11]
spm_knots     = [0.03, 0.81, 1.62, 2.36, 4.43]

def make_segments(series: pd.Series, knots: list, name: str) -> pd.DataFrame:
    """Create piece-wise segment-length columns for a given series."""
    segs = {}
    prev = 0.0
    for i, k in enumerate(knots, start=1):
        segs[f"{name}_seg{i}"] = np.clip(series - prev, 0, k - prev)
        prev = k
    segs[f"{name}_seg{len(knots)+1}"] = np.clip(series - prev, 0, None)
    return pd.DataFrame(segs)

def round_to_quarter(x: np.ndarray) -> np.ndarray:
    """
    Round array of floats to nearest $0.25 using standard half-up rounding.
    Equivalent MAE to banker's rounding for quarters in our data.
    """
    return np.round(x * 4) / 4

def load_data(filename='public_cases.json') -> pd.DataFrame:
    """Load the JSON and flatten into a DataFrame with needed columns."""
    with open(filename) as f:
        data = json.load(f)
    rows = []
    for case in data:
        inp = case['input']
        rows.append({
            'days':     inp['trip_duration_days'],
            'miles':    inp['miles_traveled'],
            'receipts': inp['total_receipts_amount'],
            'payout':   case['expected_output']
        })
    return pd.DataFrame(rows)

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create all segment features and return the feature matrix."""
    df['mpd'] = df['miles'] / df['days']
    df['spd'] = df['receipts'] / df['days']
    df['spm'] = df['receipts'] / df['miles'].replace(0, 1)
    seg_frames = [
        make_segments(df['mpd'],     mpd_knots,     'mpd'),
        make_segments(df['receipts'], receipt_knots, 'rcpt'),
        make_segments(df['spd'],     spd_knots,     'spd'),
        make_segments(df['days'],    days_knots,    'days'),
        make_segments(df['spm'],     spm_knots,     'spm'),
    ]
    feats = pd.concat(seg_frames, axis=1)
    feats['high_mpd6']    = (df['mpd']  > 894   ).astype(int)
    feats['low_rcpt4']    = (df['receipts'] <= 17.77).astype(int)
    feats['zero_spm_seg2'] = (df['spm']  <= 0     ).astype(int)
    return feats

def main():
    df = load_data()
    feats = build_features(df)
    X = feats.values
    y = df['payout'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Fit RidgeCV
    ridge = RidgeCV(alphas=[0.1, 1.0, 10.0], cv=5)
    ridge.fit(X_train, y_train)
    raw_pred = ridge.predict(X_test)
    rounded_pred = round_to_quarter(raw_pred)

    print("=== RidgeCV Performance (before rounding) ===")
    print(f"R²: {r2_score(y_test, raw_pred):.4f}")
    print(f"RMSE: ${np.sqrt(mean_squared_error(y_test, raw_pred)):.2f}")
    print(f"MAE: ${mean_absolute_error(y_test, raw_pred):.2f}\n")

    print("=== RidgeCV Performance (after quarter-rounding) ===")
    print(f"R²: {r2_score(y_test, rounded_pred):.4f}")
    print(f"RMSE: ${np.sqrt(mean_squared_error(y_test, rounded_pred)):.2f}")
    print(f"MAE: ${mean_absolute_error(y_test, rounded_pred):.2f}\n")

        # —————— new block starts here ——————
    # 1) Compute residuals on the full training set
    full_raw_pred = ridge.predict(X)     # or bst.predict(X) if you used XGB
    resid = y - full_raw_pred

    # 2) Fit an unpruned decision tree to the residuals
    from sklearn.tree import DecisionTreeRegressor, export_text
    tree = DecisionTreeRegressor(max_depth=None, min_samples_leaf=1, random_state=0)
    tree.fit(feats.values, resid)

    # 3) Print out all the “if/else” splits on your segment features
    rules = export_text(tree, feature_names=feats.columns.tolist())
    print("\n=== Surrogate Tree Residual Rules ===")
    print(rules)
    # —————— new block ends here ——————

    # Save models
    joblib.dump(ridge, 'ridge_gam_model.pkl')
    print("Saved Ridge model to ridge_gam_model.pkl")

    # Save rounding function for inference
    joblib.dump(round_to_quarter, 'round_to_quarter.pkl')
    print("Saved rounding function to round_to_quarter.pkl")

if __name__ == "__main__":
    main()
