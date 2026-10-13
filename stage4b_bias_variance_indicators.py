
from pathlib import Path
import numpy as np
import pandas as pd

# ============================================================
# Stage 4B: Bias proxies and variance indicators
# Exact column matching for existing window-feature CSV files
# ============================================================

BASE = Path(
    "/home/fatima/Research/validation/"
    "imu_degradation_results/window_feature_analysis"
)

OUT = BASE / "stage4b_bias_variance"
OUT.mkdir(parents=True, exist_ok=True)

INPUT_FILES = {
    "F2": BASE / "flight2_window_features.csv",
    "F3": BASE / "flight3_window_features.csv",
    "F5": BASE / "flight5_window_features.csv",
}

AXES = ["x", "y", "z"]
SENSORS = ["accel", "gyro"]


def numeric_column(df, name):
    """Read an exact column name as numeric data."""

    if name not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)

    return pd.to_numeric(df[name], errors="coerce")


def main():
    all_results = []
    audit_rows = []

    expected_indicators = []

    for sensor in SENSORS:
        for axis in AXES:
            prefix = f"{sensor}_{axis}"

            expected_indicators.extend([
                f"{prefix}_mean_proxy",
                f"{prefix}_std",
                f"{prefix}_variance",
                f"{prefix}_mad",
            ])

    for flight, file_path in INPUT_FILES.items():

        print("\n" + "=" * 65)
        print(f"Processing {flight}")
        print(f"Input: {file_path}")
        print("=" * 65)

        if not file_path.exists():
            print("ERROR: Input file not found.")
            audit_rows.append({
                "flight": flight,
                "status": "missing_file",
                "rows": 0,
            })
            continue

        df = pd.read_csv(file_path)

        if df.empty:
            print("ERROR: Input file is empty.")
            audit_rows.append({
                "flight": flight,
                "status": "empty_file",
                "rows": 0,
            })
            continue

        # ----------------------------------------------------
        # Time and window identifiers
        # ----------------------------------------------------

        required_time_cols = ["start_s", "end_s"]

        if not all(c in df.columns for c in required_time_cols):
            raise ValueError(
                f"{flight}: expected start_s and end_s. "
                f"Found columns: {df.columns.tolist()}"
            )

        result = pd.DataFrame(index=df.index)

        result["flight"] = flight

        if "window" in df.columns:
            result["window_id"] = df["window"]
        else:
            result["window_id"] = np.arange(len(df))

        start_s = numeric_column(df, "start_s")
        end_s = numeric_column(df, "end_s")

        result["window_start_s"] = start_s
        result["window_end_s"] = end_s
        result["window_mid_s"] = (start_s + end_s) / 2.0

        # Preserve available quality information.
        for col in [
            "duration_s",
            "n_samples",
            "median_fs_hz",
            "jitter_std_percent",
            "timestamp_quality",
        ]:
            if col in df.columns:
                result[col] = df[col]

        missing = []

        # ----------------------------------------------------
        # Exact sensor-axis-statistic columns
        # ----------------------------------------------------

        for sensor in SENSORS:
            for axis in AXES:

                prefix = f"{sensor}_{axis}"

                source_columns = {
                    "mean_proxy": f"{prefix}_mean",
                    "std": f"{prefix}_std",
                    "mad": f"{prefix}_mad",
                }

                for output_stat, source_col in source_columns.items():

                    output_col = f"{prefix}_{output_stat}"

                    if source_col not in df.columns:
                        result[output_col] = np.nan
                        missing.append(source_col)
                        continue

                    result[output_col] = numeric_column(
                        df, source_col
                    )

                # Variance calculated from the ordinary STD,
                # NOT highpass_std.
                std_col = f"{prefix}_std"
                variance_col = f"{prefix}_variance"

                if std_col in result.columns:
                    std_values = result[std_col].where(
                        result[std_col] >= 0
                    )

                    result[variance_col] = std_values ** 2
                else:
                    result[variance_col] = np.nan
                    missing.append(std_col)

        # ----------------------------------------------------
        # Validate data before saving
        # ----------------------------------------------------

        result = result.replace([np.inf, -np.inf], np.nan)

        if result["window_mid_s"].isna().any():
            raise ValueError(
                f"{flight}: missing or invalid window timestamps."
            )

        # Ensure all expected indicator columns exist.
        for col in expected_indicators:
            if col not in result.columns:
                result[col] = np.nan

        output_file = OUT / f"{flight}_stage4b_indicators.csv"
        result.to_csv(output_file, index=False)

        all_results.append(result)

        missing_unique = sorted(set(missing))

        audit_rows.append({
            "flight": flight,
            "status": "processed",
            "rows": len(result),
            "missing_columns": ";".join(missing_unique),
            "missing_indicator_count": len(missing_unique),
            "output_file": str(output_file),
        })

        print("Rows:", len(result))
        print("Time column: window_mid_s")
        print("Missing source columns:", missing_unique)
        print("Saved:", output_file)

        # Show completeness for the key statistics.
        for stat in ["mean_proxy", "std", "variance", "mad"]:
            cols = [
                f"{sensor}_{axis}_{stat}"
                for sensor in SENSORS
                for axis in AXES
            ]

            valid_counts = result[cols].notna().sum()

            print(
                f"{stat}: {int(valid_counts.sum())}/"
                f"{len(result) * len(cols)} valid values"
            )

    # --------------------------------------------------------
    # Combined data and summary
    # --------------------------------------------------------

    if not all_results:
        raise RuntimeError(
            "No flights processed successfully. "
            "Check the input files."
        )

    combined = pd.concat(all_results, ignore_index=True)

    combined_file = OUT / "F2_F3_F5_stage4b_indicators.csv"
    combined.to_csv(combined_file, index=False)

    indicator_cols = [
        c for c in expected_indicators
        if c in combined.columns
    ]

    summary_rows = []

    for flight, group in combined.groupby("flight"):

        for col in indicator_cols:

            values = pd.to_numeric(
                group[col], errors="coerce"
            ).dropna()

            summary_rows.append({
                "flight": flight,
                "indicator": col,
                "n_valid": len(values),
                "mean": values.mean() if len(values) else np.nan,
                "std_between_windows": (
                    values.std(ddof=1)
                    if len(values) > 1 else np.nan
                ),
                "median": (
                    values.median() if len(values) else np.nan
                ),
                "min": values.min() if len(values) else np.nan,
                "max": values.max() if len(values) else np.nan,
            })

    summary = pd.DataFrame(summary_rows)

    summary_file = OUT / "stage4b_indicator_summary.csv"
    summary.to_csv(summary_file, index=False)

    audit_file = OUT / "stage4b_input_audit.csv"
    pd.DataFrame(audit_rows).to_csv(audit_file, index=False)

    # --------------------------------------------------------
    # Final completeness check
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("STAGE 4B FINAL VALIDATION")
    print("=" * 65)

    print("Combined rows:", len(combined))
    print("Expected rows: 211")

    print("\nMissing values per indicator:")
    print(combined[indicator_cols].isna().sum().to_string())

    print("\nOutput files:")
    print(combined_file)
    print(summary_file)
    print(audit_file)

    print("\nStage 4B finished.")


if __name__ == "__main__":
    main()
