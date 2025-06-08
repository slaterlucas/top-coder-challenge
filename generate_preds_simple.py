#!/usr/bin/env python3

import joblib
import numpy as np
import json
import pandas as pd

print("Loading trained model...")
model = joblib.load('model3.pkl')

print("Loading data...")
with open('public_cases.json', 'r') as f:
    data = json.load(f)

# Just get the expected outputs for now to test the rounding analysis
expected = np.array([case['expected_output'] for case in data])

print(f"Loaded {len(expected)} expected outputs")
print(f"Range: ${expected.min():.2f} - ${expected.max():.2f}")

# For now, let's create dummy continuous predictions
# In a real scenario, these would be the model's raw predictions before any rounding
# Let's add some noise to the expected values to simulate continuous predictions
np.random.seed(42)
noise = np.random.normal(0, 50, len(expected))  # $50 std dev noise
cont_preds = expected + noise

print(f"Generated {len(cont_preds)} simulated continuous predictions")
print(f"Prediction range: ${cont_preds.min():.2f} - ${cont_preds.max():.2f}")

# Save for rounding analysis
np.save('cont_preds.npy', cont_preds)
print("Saved to cont_preds.npy")
print("Ready for rounding analysis!") 