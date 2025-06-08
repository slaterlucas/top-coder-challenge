#!/usr/bin/env python3

import sys
import joblib
import numpy as np
import pandas as pd

def create_advanced_features(df):
    """Feature engineering v5 – uses the new breakpoint buckets"""

    # -------- basic ratios --------
    df["miles_per_day"]  = df["miles_traveled"] / np.maximum(df["trip_duration_days"], 1)
    df["spend_per_day"]  = df["total_receipts_amount"] / np.maximum(df["trip_duration_days"], 1)
    df["spend_per_mile"] = df["total_receipts_amount"] / np.maximum(df["miles_traveled"], 1)

    # -------- piece-wise segment lengths --------
    def seg(series, *knots):
        out, prev = {}, 0
        for i, k in enumerate(knots, 1):
            out[f"{series.name}_seg{i}"] = np.clip(series - prev, 0, k - prev)
            prev = k
        out[f"{series.name}_seg{len(knots)+1}"] = np.clip(series - prev, 0, None)
        return pd.DataFrame(out)

    df = pd.concat(
        [
            df,
            seg(df["miles_per_day"], 1, 37, 67, 103, 181),         # mpd segments 6 cols
            seg(df["total_receipts_amount"], 6, 451, 923, 1406, 1800, 1992),
            seg(df["spend_per_day"], 2, 70, 135, 199, 345),
            seg(df["trip_duration_days"], 2, 4, 6, 8, 11),
        ],
        axis=1,
    )

    # -------- legacy flags that still help --------
    df["overspend_amt"]  = np.clip(df["spend_per_day"] - 120, 0, None)
    df["overspend_flag"] = (df["overspend_amt"] > 0).astype(int)

    df["excess_milespd"]    = np.clip(df["miles_per_day"] - 300, 0, None)
    df["excess_miles_flag"] = (df["excess_milespd"] > 0).astype(int)

    # -------- interactions (keep the simple ones) --------
    df["dur_x_mpd"]   = df["trip_duration_days"] * df["miles_per_day"]
    df["dur_x_spd"]   = df["trip_duration_days"] * df["spend_per_day"]
    df["rcpt_x_days"] = df["total_receipts_amount"] * df["trip_duration_days"]

    df['is_1day'] = (df['trip_duration_days'] == 1).astype(int)
    df['is_4day'] = (df['trip_duration_days'] == 4).astype(int)
    df['is_5day'] = (df['trip_duration_days'] == 5).astype(int)
    df['is_6day'] = (df['trip_duration_days'] == 6).astype(int)

    # 1) extract the cents portion
    cents = ((df["total_receipts_amount"] * 100).round().astype(int)) % 100
    # 2) binary flags
    df["ends_49c"] = (cents == 49).astype(int)
    df["ends_99c"] = (cents == 99).astype(int)

    return df

def predict_reimbursement(trip_days, miles, receipts):
    """
    Predict reimbursement amount using Model 3 with breakpoint-based features
    """
    try:
        # Load Model 4
        model = joblib.load('model4.pkl')
        
        # Create DataFrame for feature engineering
        test_df = pd.DataFrame({
            'trip_duration_days': [trip_days],
            'miles_traveled': [miles],
            'total_receipts_amount': [receipts]
        })
        
        # Apply Model 3 feature engineering (breakpoint-based)
        test_df = create_advanced_features(test_df)
        
        # Get all feature columns (same order as training)
        feature_cols = [col for col in test_df.columns if col not in ['expected_output']]
        features = test_df[feature_cols].values
        
        # Make prediction
        prediction = model.predict(features)[0]
        return prediction
        
    except Exception as e:
        print(f"Error loading model or making prediction: {e}", file=sys.stderr)
        return 0.0

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 calculate_reimbursement.py <trip_duration_days> <miles_traveled> <total_receipts_amount>")
        sys.exit(1)
    
    try:
        trip_days = int(sys.argv[1])
        miles = float(sys.argv[2])
        receipts = float(sys.argv[3])
        
        reimbursement = predict_reimbursement(trip_days, miles, receipts)
        
        # Output just the number (as required)
        print(f"{reimbursement:.2f}")
        
    except ValueError as e:
        print(f"Error parsing arguments: {e}", file=sys.stderr)
        sys.exit(1) 