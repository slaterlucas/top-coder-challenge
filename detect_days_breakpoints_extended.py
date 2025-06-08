import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import ruptures as rpt


def load_data(path="public_cases.json"):
    with open(path, "r") as f:
        data = json.load(f)
    return data


def build_dataframe(cases):
    rows = []
    for c in cases:
        inp = c.get("input", {})
        rows.append({
            "days": inp.get("trip_duration_days"),
            "miles": inp.get("miles_traveled"),
            "receipts": inp.get("total_receipts_amount"),
            "payout": c.get("expected_output"),
        })
    df = pd.DataFrame(rows)
    return df


def detect_pelt_breakpoints(df_sorted):
    """Use PELT to detect breakpoints in days vs payout"""
    series = df_sorted["payout"].values
    algo = rpt.Pelt(model="l2").fit(series)
    bkpts = algo.predict(pen=10)  # Adjust penalty as needed
    
    # Remove last index which is length of series
    breakpoint_indices = [i for i in bkpts if i < len(series)]
    breakpoint_day_values = df_sorted.loc[breakpoint_indices, "days"].tolist()
    
    print(f"PELT detected {len(breakpoint_day_values)} breakpoints:")
    for day in breakpoint_day_values:
        print(f"  {day} days")
    
    return breakpoint_day_values


def select_optimal_knots(df_sorted, candidate_knots, num_knots=5):
    """Select optimal knots using evenly spaced approach"""
    # Sort candidates and ensure they're within data range
    candidates = sorted([k for k in candidate_knots 
                        if df_sorted["days"].min() < k < df_sorted["days"].max()])
    
    if len(candidates) < num_knots:
        print(f"Warning: Only {len(candidates)} candidate knots available, using all")
        selected_knots = candidates
    else:
        # Use evenly spaced knots from candidates
        step = len(candidates) // num_knots
        selected_knots = [candidates[i*step] for i in range(num_knots)]
    
    return selected_knots


def fit_piecewise_model(df_sorted, knots):
    """Fit piecewise linear model with given knots"""
    days = df_sorted["days"].values
    payout = df_sorted["payout"].values
    
    # Create feature matrix
    X = np.ones((len(days), 1))  # intercept
    for k in knots:
        X = np.hstack([X, np.maximum(0, days.reshape(-1, 1) - k)])
    
    # Fit model
    model = LinearRegression(fit_intercept=False)
    model.fit(X, payout)
    
    # Calculate metrics
    preds = model.predict(X)
    r2 = model.score(X, payout)
    mae = np.mean(np.abs(payout - preds))
    rmse = np.sqrt(np.mean((payout - preds) ** 2))
    
    print(f"\nModel performance:")
    print(f"  R²: {r2:.4f}")
    print(f"  MAE: ${mae:.2f}")
    print(f"  RMSE: ${rmse:.2f}")
    
    # Print segment information
    coeffs = model.coef_
    print(f"\nPiecewise model segments:")
    print(f"  Base intercept: ${coeffs[0]:.2f}")
    
    cumulative_slope = 0
    for i, (knot, coeff) in enumerate(zip(knots, coeffs[1:])):
        cumulative_slope += coeff
        print(f"  Above {knot} days: slope = {cumulative_slope:.3f}")
    
    return model, r2, mae, rmse


def create_visualization(df_sorted, knots, model, filename="days_breakpoints_analysis.png"):
    """Create visualization of the piecewise model"""
    days_range = np.linspace(df_sorted["days"].min(), 
                            df_sorted["days"].max(), 1000)
    
    # Create prediction for the range
    X_pred = np.ones((len(days_range), 1))
    for k in knots:
        X_pred = np.hstack([X_pred, np.maximum(0, days_range.reshape(-1, 1) - k)])
    
    payout_pred = model.predict(X_pred)
    
    plt.figure(figsize=(14, 10))
    plt.scatter(df_sorted["days"], df_sorted["payout"], alpha=0.6, s=20, label="Training Data")
    plt.plot(days_range, payout_pred, 'r-', linewidth=3, label=f"{len(knots)}-knot model")
    
    # Mark the knots
    for i, k in enumerate(knots):
        plt.axvline(k, color='orange', linestyle='--', alpha=0.8, linewidth=2)
        plt.text(k, plt.ylim()[1]*0.95, f'{k}d', rotation=90, ha='right', fontsize=10, fontweight='bold')
    
    plt.xlabel("Trip Duration (days)", fontsize=12)
    plt.ylabel("Payout ($)", fontsize=12)
    plt.title(f"Trip Duration vs Payout - {len(knots)}-Knot Piecewise Model (All {len(df_sorted)} cases)", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {filename}")


def main():
    cases = load_data()
    df = build_dataframe(cases)
    
    print(f"=== TRIP DURATION BREAKPOINT ANALYSIS ===")
    print(f"Analyzing all {len(df)} cases")
    print(f"Days range: {df['days'].min()} - {df['days'].max()} days")
    print(f"Payout range: ${df['payout'].min():.2f} - ${df['payout'].max():.2f}")
    
    # Sort by days for analysis
    df_sorted = df.sort_values("days").reset_index(drop=True)
    
    # Detect breakpoints using PELT
    pelt_breakpoints = detect_pelt_breakpoints(df_sorted)
    
    # Select 5 optimal knots
    optimal_knots = select_optimal_knots(df_sorted, pelt_breakpoints, num_knots=5)
    print(f"\nSelected 5 knots: {optimal_knots}")
    
    # Fit piecewise model
    model, r2, mae, rmse = fit_piecewise_model(df_sorted, optimal_knots)
    
    # Save results
    results = {
        "knots": optimal_knots,
        "pelt_breakpoints": pelt_breakpoints,
        "num_cases": len(df),
        "days_range": [int(df["days"].min()), int(df["days"].max())],
        "model_performance": {
            "r2": r2,
            "mae": mae,
            "rmse": rmse
        },
        "coefficients": model.coef_.tolist()
    }
    
    with open("days_5knots_analysis.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: days_5knots_analysis.json")
    
    # Create visualization
    create_visualization(df_sorted, optimal_knots, model)


if __name__ == "__main__":
    main() 