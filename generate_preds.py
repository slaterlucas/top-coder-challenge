#!/usr/bin/env python3

import joblib
import numpy as np
import json
from model4 import create_advanced_features
import pandas as pd

# Load the trained model
print("Loading trained model...")
model = joblib.load('model3.pkl')

# Load data
print("Loading public cases...")
with open('public_cases.json', 'r') as f:
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
print(f"Loaded {len(df)} cases")

# Apply feature engineering
print("Applying feature engineering...")
df = create_advanced_features(df)

# Prepare feature matrix (excluding target)
feature_cols = [col for col in df.columns if col != 'expected_output']
X = df[feature_cols].values

print(f"Feature matrix shape: {X.shape}")

# Generate predictions
print("Generating continuous predictions...")
cont_preds = model.predict(X)

# Save for rounding analysis
np.save('cont_preds.npy', cont_preds)

print(f"Generated {len(cont_preds)} continuous predictions")
print(f"Prediction range: ${cont_preds.min():.2f} - ${cont_preds.max():.2f}")
print(f"Mean prediction: ${cont_preds.mean():.2f}")
print("Saved to cont_preds.npy")
print("Ready for rounding analysis!") 