#!/usr/bin/env python3

import xgboost as xgb
import pandas as pd
import numpy as np
import json
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Score

def create_advanced_features(df):
    """Create sophisticated feature engineering based on business logic patterns"""
    
    # Basic ratios
    df['miles_per_day'] = df['miles_traveled'] / np.maximum(df['trip_duration_days'], 1)
    df['spend_per_day'] = df['total_receipts_amount'] / np.maximum(df['trip_duration_days'], 1) 
    df['spend_per_mile'] = df['total_receipts_amount'] / np.maximum(df['miles_traveled'], 1)
    
    # 1. Dynamic spend ceiling (cap rises then falls with duration)
    def spend_cap(d):
        if d <= 3:  return 75          # short trip ceiling
        if d <= 6:  return 120         # 4–6-day sweet spot
        return 90                      # 7+-day ceiling
    
    df["spend_cap"] = df["trip_duration_days"].apply(spend_cap)
    df["overspend_amt"] = (df["spend_per_day"] - df["spend_cap"]).clip(lower=0)
    df["overspend_flag"] = (df["overspend_amt"] > 0).astype(int)
    
    # 2. High-mileage penalty
    df["miles_cap"] = 300
    df["excess_milespd"] = (df["miles_per_day"] - 300).clip(lower=0)
    df["excess_miles_flag"] = (df["excess_milespd"] > 0).astype(int)
    
    # 3. Clip + keep raw
    df["receipts_clipped"] = df["total_receipts_amount"].clip(upper=800)
    df["receipts_excess"] = (df["total_receipts_amount"] - 800).clip(lower=0)
    
    # Categorical bins based on business rules - ensure full coverage
    df['receipt_band'] = pd.cut(df['total_receipts_amount'], 
                               bins=[0, 50, 600, 800, float('inf')], 
                               labels=[0, 1, 2, 3], 
                               include_lowest=True).fillna(0).astype(int)
    
    df['duration_bin'] = pd.cut(df['trip_duration_days'],
                               bins=[0, 3, 6, 9, float('inf')],
                               labels=[0, 1, 2, 3],
                               include_lowest=True).fillna(0).astype(int)
    
    df['mileage_tier'] = pd.cut(df['miles_traveled'],
                               bins=[0, 100, 600, 800, float('inf')],
                               labels=[0, 1, 2, 3], 
                               include_lowest=True).fillna(0).astype(int)
    
    # Efficiency band (sweet spot around 180-220 mi/day)
    df['efficiency_band'] = pd.cut(df['miles_per_day'],
                                  bins=[0, 100, 180, 220, 300, float('inf')],
                                  labels=[0, 1, 2, 3, 4],
                                  include_lowest=True).fillna(0).astype(int)
    
    # Interaction terms
    df['dur_x_eff'] = df['trip_duration_days'] * df['miles_per_day']
    df['dur_x_spendpd'] = df['trip_duration_days'] * df['spend_per_day']
    df['mileage_x_efficiency'] = df['mileage_tier'] * df['efficiency_band']
    df['receipt_x_duration'] = df['receipt_band'] * df['duration_bin']
    
    # Additional business logic features
    df['is_efficient_trip'] = ((df['miles_per_day'] >= 180) & (df['miles_per_day'] <= 220)).astype(int)
    df['is_sweet_spot_receipts'] = ((df['total_receipts_amount'] >= 600) & (df['total_receipts_amount'] <= 800)).astype(int)
    df['is_moderate_duration'] = ((df['trip_duration_days'] >= 4) & (df['trip_duration_days'] <= 6)).astype(int)
    
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

print(f"\nTraining samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Advanced hyperparameter search for rule-based system
print(f"\nTraining advanced model...")
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [6, 8, 10, 12],  # Deeper trees for rule capture
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'reg_alpha': [0, 0.1, 0.5],
    'reg_lambda': [0.1, 0.5, 1.0],
    'min_child_weight': [1, 3, 5]
}

grid_search = GridSearchCV(
    xgb.XGBRegressor(random_state=42),
    param_grid,
    cv=5,
    scoring='neg_mean_absolute_error',  # Focus on MAE for precision
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)
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
joblib.dump(best_model, 'model3.pkl')
print(f"\n✅ Model 3 saved as 'model3.pkl'")

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