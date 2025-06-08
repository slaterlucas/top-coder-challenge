#!/bin/bash

# Black Box Challenge - XGBoost Implementation
# This script uses a trained XGBoost model to predict reimbursement amounts
# Usage: ./run.sh <trip_duration_days> <miles_traveled> <total_receipts_amount>

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Use the trained XGBoost model
python3 calculate_reimbursement.py "$1" "$2" "$3" 