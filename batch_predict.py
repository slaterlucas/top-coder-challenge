#!/usr/bin/env python3

import sys
import joblib
import numpy as np
import pandas as pd
import json

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

if __name__ == "__main__":
    # Load model once
    print("Loading model...", file=sys.stderr)
    model = joblib.load('model4.pkl')
    
    # Load all private cases
    print("Loading private cases...", file=sys.stderr)
    with open('private_cases.json', 'r') as f:
        data = json.load(f)
    
    # Convert to DataFrame
    rows = []
    for case in data:
        rows.append({
            'trip_duration_days': case['trip_duration_days'],
            'miles_traveled': case['miles_traveled'],
            'total_receipts_amount': case['total_receipts_amount']
        })
    
    df = pd.DataFrame(rows)
    print(f"Processing {len(df)} cases...", file=sys.stderr)
    
    # Apply feature engineering to all cases at once
    df = create_advanced_features(df)
    
    # Get feature columns
    feature_cols = [col for col in df.columns if col not in ['expected_output']]
    X = df[feature_cols].values
    
    # Predict all at once
    print("Making predictions...", file=sys.stderr)
    predictions = model.predict(X)
    
    # Output results
    for pred in predictions:
        print(f"{pred:.2f}")
    
    print("Done!", file=sys.stderr) 