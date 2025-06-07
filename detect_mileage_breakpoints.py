import argparse
import json
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import ruptures as rpt
import matplotlib.pyplot as plt


def load_cases(path: str) -> pd.DataFrame:
    """Load JSON cases file into DataFrame."""
    with open(path, "r") as f:
        data = json.load(f)
    records = []
    for case in data:
        inp = case.get("input", {})
        records.append(
            {
                "days": inp.get("trip_duration_days"),
                "miles": inp.get("miles_traveled"),
                "receipts": inp.get("total_receipts_amount"),
                "payout": case.get("expected_output"),
            }
        )
    return pd.DataFrame(records)


def filter_cases(
    df: pd.DataFrame,
    days_min: int,
    days_max: int,
    receipts_min: float,
    receipts_max: float,
) -> pd.DataFrame:
    mask = (
        (df["days"] >= days_min)
        & (df["days"] <= days_max)
        & (df["receipts"] >= receipts_min)
        & (df["receipts"] <= receipts_max)
    )
    return df.loc[mask].copy()


def compute_local_slope(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("miles").reset_index(drop=True)
    slopes = np.gradient(df["payout"], df["miles"])
    df["local_slope"] = slopes
    return df


def run_pelt_breakpoints(values: np.ndarray) -> List[int]:
    algo = rpt.Pelt(model="l2").fit(values)
    idx = algo.predict(pen=10)
    # omit last index equal to len(values)
    return idx[:-1]


def add_mileage_segments(df: pd.DataFrame, k1: float, k2: Optional[float] = None) -> pd.DataFrame:
    """Add piecewise linear segment columns based on knots."""
    m = df["miles"].values
    seg1 = np.minimum(m, k1)
    if k2 is None:
        seg2 = np.maximum(0, m - k1)
        seg3 = np.zeros_like(m)
    else:
        seg2 = np.maximum(0, np.minimum(m, k2) - k1)
        seg3 = np.maximum(0, m - k2)
    df = df.copy()
    df["miles_seg1"] = seg1
    df["miles_seg2"] = seg2
    df["miles_seg3"] = seg3
    return df


def fit_hinge(df: pd.DataFrame, k1: float, k2: Optional[float] = None) -> Tuple[float, LinearRegression]:
    df_seg = add_mileage_segments(df, k1, k2)
    features = ["miles_seg1", "miles_seg2"] if k2 is None else ["miles_seg1", "miles_seg2", "miles_seg3"]
    X = df_seg[features].values
    y = df_seg["payout"].values
    model = LinearRegression().fit(X, y)
    pred = model.predict(X)
    rss = np.sum((y - pred) ** 2)
    n = len(y)
    p = X.shape[1] + 1
    aic = n * np.log(rss / n) + 2 * p
    return aic, model


def choose_one_knot(df: pd.DataFrame, candidates: List[int]) -> Tuple[int, LinearRegression]:
    best_aic = float("inf")
    best_k = candidates[0]
    best_model = None
    for k in candidates:
        aic, model = fit_hinge(df, k)
        if aic < best_aic:
            best_aic = aic
            best_k = k
            best_model = model
    return best_k, best_model


def choose_two_knots(df: pd.DataFrame, candidates: List[int]) -> Tuple[Tuple[int, int], LinearRegression]:
    pairs = [(c1, c2) for c1 in candidates for c2 in candidates if c1 < c2]
    best_aic = float("inf")
    best_pair = pairs[0]
    best_model = None
    for k1, k2 in pairs:
        aic, model = fit_hinge(df, k1, k2)
        if aic < best_aic:
            best_aic = aic
            best_pair = (k1, k2)
            best_model = model
    return best_pair, best_model


def save_plots(df: pd.DataFrame, output_prefix: str) -> None:
    plt.figure(figsize=(8, 6))
    plt.scatter(df["miles"], df["payout"], alpha=0.6)
    plt.xlabel("Miles")
    plt.ylabel("Payout")
    plt.title("Miles vs Payout")
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_scatter.png")
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(df["miles"], df["local_slope"], marker="o", linestyle="-")
    plt.xlabel("Miles")
    plt.ylabel("Local slope")
    plt.title("Local Slope of Payout vs Miles")
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_slope.png")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect mileage breakpoints")
    parser.add_argument("cases", nargs="?", default="public_cases.json")
    parser.add_argument("--days-min", type=int, default=4)
    parser.add_argument("--days-max", type=int, default=6)
    parser.add_argument("--receipts-min", type=float, default=550)
    parser.add_argument("--receipts-max", type=float, default=650)
    parser.add_argument("--two-knots", action="store_true")
    args = parser.parse_args()

    df = load_cases(args.cases)
    df = filter_cases(df, args.days_min, args.days_max, args.receipts_min, args.receipts_max)
    df = compute_local_slope(df)

    break_idx = run_pelt_breakpoints(df["payout"].values)
    break_miles = df.loc[break_idx, "miles"].tolist()

    candidates = [50, 100, 150, 200, 400, 600, 800, 1000]

    if args.two_knots:
        (k1, k2), model = choose_two_knots(df, candidates)
    else:
        k1, model = choose_one_knot(df, candidates)
        k2 = None

    seg_df = add_mileage_segments(df, k1, k2)
    features = ["miles_seg1", "miles_seg2"] if k2 is None else ["miles_seg1", "miles_seg2", "miles_seg3"]
    X = seg_df[features].values
    y = seg_df["payout"].values
    pred = model.predict(X)

    if k2 is None:
        slopes = [model.coef_[0], model.coef_[0] + model.coef_[1]]
    else:
        slopes = [model.coef_[0], model.coef_[0] + model.coef_[1], model.coef_[0] + model.coef_[1] + model.coef_[2]]

    result = {
        "knot_1": int(k1),
        "knot_2": int(k2) if k2 is not None else None,
        "slope_1": float(slopes[0]),
        "slope_2": float(slopes[1]),
    }
    if k2 is not None:
        result["slope_3"] = float(slopes[2])
    with open("mileage_segments.json", "w") as f:
        json.dump(result, f, indent=2)

    print("Detected breakpoints from PELT:", break_miles)
    print("Selected knot(s):", k1 if k2 is None else (k1, k2))
    print("Slopes:", slopes)

    df["fitted"] = pred
    save_plots(df, "mileage")


if __name__ == "__main__":
    main()
