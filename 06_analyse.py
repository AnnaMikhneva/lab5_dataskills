
import json
import yaml
import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def load_params(params_path: str = "params.yaml") -> dict:
    with open(params_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Produce summary CSV")
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()

    load_params(args.params)

    distances_dir = Path("data/distances")
    features_dir  = Path("data/features")
    reports_dir   = Path("data/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Load results from upstream stages
    summary_df = pd.read_csv(distances_dir / "summary_stats.csv")

    with open(distances_dir / "timing.json") as f:
        timing = json.load(f)

    with open(features_dir / "precision_sizes.json") as f:
        sizes = json.load(f)

    # Enrich with size and timing columns
    summary_df["size_mb"] = summary_df["format"].map(sizes)
    summary_df["time_s"]  = summary_df["format"].map(timing)

    # Save
    out_path = reports_dir / "summary.csv"
    summary_df.to_csv(out_path, index=False)

    # Print to console
    print("\n=== Summary ===")
    print(summary_df[[
        "format", "mean_intra", "mean_inter", "ratio",
        "mean_dev_from_f64", "size_mb", "time_s"
    ]].to_string(index=False, float_format="{:.5f}".format))
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()