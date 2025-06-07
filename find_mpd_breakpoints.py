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
    df["miles_per_day"] = df["miles"] / df["days"]
    return df


def filter_dataframe(df):
    mask = (
        (df["days"] >= 4)
        & (df["days"] <= 6)
        & (df["receipts"] >= 550)
        & (df["receipts"] <= 650)
    )
    return df[mask].copy()


def compute_slope(df):
    df_sorted = df.sort_values("miles_per_day").reset_index(drop=True)
    diff_payout = df_sorted["payout"].diff()
    diff_mpd = df_sorted["miles_per_day"].diff()
    df_sorted["slope"] = diff_payout / diff_mpd
    return df_sorted


def detect_breakpoints(df_sorted):
    series = df_sorted["payout"].values
    algo = rpt.Pelt(model="l2").fit(series)
    bkpts = algo.predict(pen=10)
    # Remove last index which is length of series
    idx = [i for i in bkpts if i < len(series)]
    knots = df_sorted.loc[np.array(idx), "miles_per_day"].tolist()
    print("Candidate breakpoints from ruptures:")
    for k in knots:
        print(f"  {k:.2f}")
    return knots


def hinge_regression_aic(df_sorted, candidate_knots):
    mpd = df_sorted["miles_per_day"].values.reshape(-1, 1)
    payout = df_sorted["payout"].values
    best_knot = None
    best_aic = np.inf
    best_params = None
    n = len(df_sorted)
    for k in candidate_knots:
        X = np.hstack([
            np.minimum(mpd, k),
            np.maximum(0, mpd - k),
        ])
        model = LinearRegression(fit_intercept=True)
        model.fit(X, payout)
        preds = model.predict(X)
        rss = np.sum((payout - preds) ** 2)
        k_params = 3  # intercept + two slopes
        aic = n * np.log(rss / n) + 2 * k_params
        print(f"knot {k}: AIC={aic:.2f}")
        if aic < best_aic:
            best_aic = aic
            best_knot = k
            coeffs = np.concatenate(([model.intercept_], model.coef_))
            best_params = coeffs
    b0, b1, b2 = best_params
    print(f"Best knot: {best_knot} with AIC={best_aic:.2f}")
    return best_knot, b1, b2


def main():
    cases = load_data()
    df = build_dataframe(cases)
    df_filtered = filter_dataframe(df)

    if df_filtered.empty:
        print("No data after filtering")
        return

    df_slope = compute_slope(df_filtered)

    # Scatter plot
    plt.figure()
    plt.scatter(df_slope["miles_per_day"], df_slope["payout"], alpha=0.7)
    plt.xlabel("Miles per Day")
    plt.ylabel("Payout")
    plt.title("Miles per Day vs Payout")
    plt.savefig("scatter_mpd_vs_payout.png")

    # Slope plot
    plt.figure()
    plt.plot(df_slope["miles_per_day"], df_slope["slope"], marker="o")
    plt.xlabel("Miles per Day")
    plt.ylabel("Local Slope Δpayout/Δmpd")
    plt.title("Local Slope of Payout vs Miles per Day")
    plt.savefig("slope_mpd.png")

    detect_breakpoints(df_slope)

    candidate_knots = [100, 150, 180, 200, 220, 250, 300, 400, 600]
    best_k, s1, s2 = hinge_regression_aic(df_slope, candidate_knots)

    with open("best_mpd_knot.json", "w") as f:
        json.dump({"knot": best_k, "slope1": s1, "slope2": s2}, f, indent=2)


if __name__ == "__main__":
    main()

