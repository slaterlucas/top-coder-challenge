#!/usr/bin/env python3

import xgboost as xgb
import pandas as pd
import numpy as np
import json
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

#

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
            seg(df["total_receipts_amount"], 6, 451, 923, 1406, 1800,1992),
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

def load_and_engineer_data(filename='public_cases.json'):
    """Load data and apply advanced feature engineering"""
    with open(filename, 'r') as f:
        data = json.load(f)
    
    # Convert to DataFrame
    rows = []
    for case in data:
        row = {
            'trip_duration_days': case['input']['trip_duration_days'],
            'miles_traveled': case['input']['miles_traveled'], 
            'total_receipts_amount': case['input']['total_receipts_amount'],
            'expected_output': case['expected_output']
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Apply feature engineering
    df = create_advanced_features(df)
    
    # Prepare feature matrix (excluding target)
    feature_cols = [col for col in df.columns if col != 'expected_output']
    X = df[feature_cols].values
    y = df['expected_output'].values
    
    return X, y, feature_cols, df

print("Loading data with advanced feature engineering...")
X, y, feature_names, df = load_and_engineer_data()

print(f"Dataset shape: {X.shape}")
print(f"Features created: {len(feature_names)}")
print("\nFeature list:")
for i, name in enumerate(feature_names):
    print(f"  {i+1:2d}. {name}")

print(f"\nData summary:")
print(df.describe())

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.10, random_state=42, shuffle=True
)

base_model = xgb.XGBRegressor(
    random_state    = 42,
    objective       = "reg:squarederror",
    n_estimators    = 2000,        # high ceiling – will stop long before this
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Advanced hyperparameter search for rule-based system
print(f"\nTraining advanced model...")
param_grid = {
    "max_depth":          [6, 8, 10],
    "learning_rate":      [0.03, 0.05, 0.1],
    "subsample":          [0.7, 0.9, 1.0],
    "colsample_bytree":   [0.5, 0.8, 1.0],
    "reg_alpha":          [0, 0.1, 0.5],
    "reg_lambda":         [0.1, 0.5, 1.0],
    "min_child_weight":   [1, 3, 5],
}

grid_search = GridSearchCV(
    estimator  = base_model,
    param_grid = param_grid,
    cv         = 3,
    scoring    = "neg_mean_absolute_error",
    n_jobs     = -1,
    verbose    = 1,
)

grid_search.fit(X_tr, y_tr)

best_model = grid_search.best_estimator_

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV MAE: {-grid_search.best_score_:.2f}")

# Evaluate model
y_pred = best_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print(f"\nAdvanced Model Performance:")
print(f"Test R²: {r2:.4f}")
print(f"Test RMSE: ${rmse:.2f}")
print(f"Test MAE: ${mae:.2f}")

# Check precision metrics
errors = np.abs(y_test - y_pred)
exact_matches = np.sum(errors <= 0.01)
close_matches = np.sum(errors <= 1.0)

print(f"\nPrecision Analysis:")
print(f"Exact matches (±$0.01): {exact_matches}/{len(y_test)} ({100*exact_matches/len(y_test):.1f}%)")
print(f"Close matches (±$1.00): {close_matches}/{len(y_test)} ({100*close_matches/len(y_test):.1f}%)")
print(f"Max error: ${np.max(errors):.2f}")

# Feature importance
print(f"\nFeature Importance:")
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

for idx, row in importance_df.head(10).iterrows():
    print(f"  {row['feature']}: {row['importance']:.4f}")

# Save model3
joblib.dump(best_model, 'model4.pkl')
print(f"\n✅ Model 4 saved as 'model4.pkl'")

# Test function
def predict_advanced(trip_days, miles, receipts):
    """Predict using advanced model with same feature engineering"""
    # Create DataFrame for feature engineering
    test_df = pd.DataFrame({
        'trip_duration_days': [trip_days],
        'miles_traveled': [miles],
        'total_receipts_amount': [receipts]
    })
    
    # Apply same feature engineering
    test_df = create_advanced_features(test_df)
    
    # Get features in same order
    test_X = test_df[[col for col in feature_names]].values
    
    prediction = best_model.predict(test_X)[0]
    return prediction

# Test the advanced model
print(f"\nTesting advanced model:")
test_cases = [
    (11, 741, 1872.39, 1847.08),
    (14, 600, 1120.05, 1847.84), 
    (8, 817, 1455.73, 1847.26)
]

for trip_days, miles, receipts, expected in test_cases:
    predicted = predict_advanced(trip_days, miles, receipts)
    error = abs(predicted - expected)
    print(f"  {trip_days} days, {miles} miles, ${receipts} → Predicted: ${predicted:.2f}, Expected: ${expected:.2f}, Error: ${error:.2f}") 