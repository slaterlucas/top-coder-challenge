import argparse
import json
from typing import List, Dict, Any


def load_cases(path: str) -> List[Dict[str, Any]]:
    """Load JSON data from the provided path."""
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def get_field(case: Dict[str, Any], key: str):
    """Return the value for key either directly or under an "input" sub-dict."""
    if key in case:
        return case[key]
    if 'input' in case and isinstance(case['input'], dict) and key in case['input']:
        return case['input'][key]
    raise KeyError(f"Key {key} not found in case")


def compute_extremes(cases: List[Dict[str, Any]], field: str):
    values = [get_field(c, field) for c in cases]
    min_val = min(values)
    max_val = max(values)
    min_cases = [c for c in cases if get_field(c, field) == min_val]
    max_cases = [c for c in cases if get_field(c, field) == max_val]
    return min_val, max_val, min_cases, max_cases


def main():
    parser = argparse.ArgumentParser(description="Find extreme values in cases")
    parser.add_argument(
        "path",
        nargs="?",
        default="private_cases.json",
        help="Path to JSON cases file (default: private_cases.json)",
    )
    parser.add_argument(
        "-n",
        "--num",
        type=int,
        default=0,
        help="Number of extreme cases to show for each field",
    )
    args = parser.parse_args()

    cases = load_cases(args.path)
    fields = ["trip_duration_days", "miles_traveled", "total_receipts_amount"]
    for field in fields:
        min_val, max_val, min_cases, max_cases = compute_extremes(cases, field)
        print(f"{field}:")
        print(f"  min: {min_val}")
        print(f"  max: {max_val}")
        if args.num > 0:
            print("  first cases with min value:")
            for case in min_cases[: args.num]:
                print(f"    {case}")
            print("  first cases with max value:")
            for case in max_cases[: args.num]:
                print(f"    {case}")
        print()


if __name__ == "__main__":
    main()
