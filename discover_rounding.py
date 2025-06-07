import json
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ALLOWED_BASES = [0.01, 0.05, 0.25, 0.50, 1.0]
ALLOWED_METHODS = ["floor", "ceil", "round", "banker"]


def load_cases(path: str) -> pd.DataFrame:
    """Load JSON cases file into a DataFrame."""
    with open(path, "r") as f:
        data = json.load(f)
    records = []
    for case in data:
        expected = case.get("expected_output")
        records.append({"expected": expected})
    return pd.DataFrame(records)


def load_preds(path: str) -> np.ndarray:
    """Load numpy array of continuous predictions."""
    return np.load(path)


def compute_frac_and_resid(expected: np.ndarray, cont_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return fractional part of expected and residuals."""
    frac = np.mod(expected, 1.0)
    resid = expected - cont_pred
    return frac, resid


def plot_fraction_hist(frac: np.ndarray, out_path: str) -> None:
    """Save histogram of fractional cents values."""
    plt.figure(figsize=(8, 6))
    bins = np.arange(0, 1.01, 0.01)
    plt.hist(frac, bins=bins, edgecolor="black")
    plt.xlabel("Fractional Part")
    plt.ylabel("Frequency")
    plt.title("Distribution of Fractional Cents")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def apply_round(x: np.ndarray, base: float, method: str) -> np.ndarray:
    """Round array x to the given base using the specified method."""
    if base not in ALLOWED_BASES:
        raise ValueError(f"Unsupported base {base}")
    if method not in ALLOWED_METHODS:
        raise ValueError(f"Unsupported method {method}")

    scaled = x / base

    if method == "floor":
        rounded = np.floor(scaled)
    elif method == "ceil":
        rounded = np.ceil(scaled)
    elif method == "round":
        # half away from zero
        rounded = np.sign(scaled) * np.floor(np.abs(scaled) + 0.5)
    else:  # banker
        rounded = np.round(scaled)

    return rounded * base


def evaluate_rounding(expected: np.ndarray, cont_pred: np.ndarray) -> pd.DataFrame:
    """Compute MAE for each rounding scheme."""
    results: List[Tuple[str, float, str]] = []
    for base in ALLOWED_BASES:
        for method in ALLOWED_METHODS:
            rounded = apply_round(cont_pred, base, method)
            mae = np.mean(np.abs(expected - rounded))
            results.append((method, base, mae))
    df = pd.DataFrame(results, columns=["method", "base", "mae"])
    df = df.sort_values("mae")
    return df


def main() -> None:
    cases_df = load_cases("public_cases.json")
    expected = cases_df["expected"].values
    cont_pred = load_preds("cont_preds.npy")

    frac, resid = compute_frac_and_resid(expected, cont_pred)
    plot_fraction_hist(frac, "rounding_frac_hist.png")

    mae_df = evaluate_rounding(expected, cont_pred)
    mae_df.to_csv("rounding_mae.csv", index=False)

    print("Top 3 rounding schemes by MAE:")
    print(mae_df.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
