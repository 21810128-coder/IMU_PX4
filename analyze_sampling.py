#!/usr/bin/env python3

"""
PX4 IMU Sampling-Rate and Data-Quality Analysis
================================================

Analyzes PX4 CSV exports from:

    ~/Research/validation/
        flight2/PX4/CSV/
        flight3/PX4/CSV/
        flight4/PX4/CSV/

Main objectives:
    1. Identify actual logging/sample rates.
    2. Detect timestamp duplicates and non-monotonic samples.
    3. Quantify timestamp jitter.
    4. Quantify data gaps using several thresholds.
    5. Compare PX4 IMU-related topics.
    6. Identify the best candidate for high-rate IMU analysis.
    7. Export CSV and Excel summaries.
    8. Produce detailed interval statistics.

IMPORTANT:
    This script does NOT resample the data.
    It only determines the actual sampling characteristics.

Author: Research workflow
"""

from pathlib import Path
import re
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path.home() / "Research" / "validation"

FLIGHT_DIRS = [
    BASE_DIR / "flight2" / "PX4" / "CSV",
    BASE_DIR / "flight3" / "PX4" / "CSV",
    BASE_DIR / "flight4" / "PX4" / "CSV",
]

OUTPUT_DIR = BASE_DIR / "analysis_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Candidate PX4 topics
PATTERNS = [
    "vehicle_imu",
    "sensor_combined",
    "sensor_accel",
    "sensor_gyro",
    "sensor_gyro_fft",
]

# Exclude status topics from sampling-rate analysis
EXCLUDE_PATTERNS = [
    "vehicle_imu_status",
]

# Timestamp units:
# PX4 ULog CSV timestamps are normally microseconds.
TIMESTAMP_SCALE = 1e-6

# Minimum number of samples needed for reliable analysis
MIN_SAMPLES = 20


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def infer_flight(filepath: Path) -> str:
    """
    Correctly identify flight number from the path.
    """
    for part in filepath.parts:
        if re.fullmatch(r"flight\d+", part.lower()):
            return part.lower()

    return "unknown"


def classify_topic(filename: str) -> str:
    """
    Classify the CSV according to PX4 topic.
    """

    name = filename.lower()

    if "vehicle_imu_status" in name:
        return "vehicle_imu_status"

    if "vehicle_imu" in name:
        return "vehicle_imu"

    if "sensor_combined" in name:
        return "sensor_combined"

    if "sensor_accel" in name:
        return "sensor_accel"

    if "sensor_gyro_fft" in name:
        return "sensor_gyro_fft"

    if "sensor_gyro" in name:
        return "sensor_gyro"

    return "other"


def find_timestamp_column(df: pd.DataFrame):
    """
    Find the timestamp column.
    """

    candidates = [
        "timestamp",
        "timestamp_sample",
        "time",
        "time_usec",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    # Fallback: search case-insensitively
    lower_map = {str(c).lower(): c for c in df.columns}

    for col in candidates:
        if col.lower() in lower_map:
            return lower_map[col.lower()]

    return None


def safe_float(value):
    """
    Convert value to float safely.
    """
    try:
        if pd.isna(value):
            return np.nan
        return float(value)
    except Exception:
        return np.nan


# ============================================================
# TIMESTAMP ANALYSIS
# ============================================================

def analyze_timestamps(df: pd.DataFrame, timestamp_column: str):
    """
    Analyze timestamp intervals.
    """

    result = {}

    ts = pd.to_numeric(
        df[timestamp_column],
        errors="coerce"
    ).dropna()

    result["timestamp_column"] = timestamp_column
    result["timestamp_valid_samples"] = int(len(ts))

    if len(ts) < MIN_SAMPLES:
        result["status"] = "INSUFFICIENT_DATA"
        return result

    # Convert microseconds -> seconds
    t = ts.to_numpy(dtype=float) * TIMESTAMP_SCALE

    # Basic timestamp properties
    dt = np.diff(t)

    result["first_timestamp_s"] = float(t[0])
    result["last_timestamp_s"] = float(t[-1])

    result["duration_s"] = float(t[-1] - t[0])
    result["duration_min"] = float((t[-1] - t[0]) / 60.0)

    # --------------------------------------------------------
    # Timestamp validity
    # --------------------------------------------------------

    result["duplicate_timestamps"] = int(np.sum(dt == 0))
    result["negative_dt"] = int(np.sum(dt < 0))
    result["positive_dt"] = int(np.sum(dt > 0))

    valid_dt = dt[dt > 0]

    if len(valid_dt) == 0:
        result["status"] = "INVALID_TIMESTAMPS"
        return result

    # --------------------------------------------------------
    # Sampling interval
    # --------------------------------------------------------

    median_dt = np.median(valid_dt)
    mean_dt = np.mean(valid_dt)

    result["dt_mean_s"] = float(mean_dt)
    result["dt_median_s"] = float(median_dt)
    result["dt_std_s"] = float(np.std(valid_dt))

    result["dt_min_s"] = float(np.min(valid_dt))
    result["dt_max_s"] = float(np.max(valid_dt))

    result["dt_mad_s"] = float(
        np.median(np.abs(valid_dt - median_dt))
    )

    # Percentiles
    percentiles = [
        50,
        90,
        95,
        99,
        99.9,
        99.99,
    ]

    for p in percentiles:
        key = f"dt_p{str(p).replace('.', '_')}_s"
        result[key] = float(np.percentile(valid_dt, p))

    # --------------------------------------------------------
    # Sampling frequency
    # --------------------------------------------------------

    result["rate_mean_hz"] = float(1.0 / mean_dt)
    result["rate_median_hz"] = float(1.0 / median_dt)

    # Robust MAD-based rate estimate
    robust_dt = median_dt

    result["rate_robust_hz"] = float(
        1.0 / robust_dt
    )

    # --------------------------------------------------------
    # Jitter
    # --------------------------------------------------------

    result["jitter_std_percent"] = float(
        100.0 * np.std(valid_dt) / median_dt
    )

    result["jitter_mad_percent"] = float(
        100.0 * result["dt_mad_s"] / median_dt
    )

    result["dt_cv_percent"] = float(
        100.0 * np.std(valid_dt) / np.mean(valid_dt)
    )

    # --------------------------------------------------------
    # Gap analysis
    # --------------------------------------------------------

    expected_dt = median_dt

    result["expected_dt_s"] = float(expected_dt)

    for factor in [1.5, 2, 3, 5, 10, 20]:
        threshold = factor * expected_dt

        result[f"gaps_gt_{factor}x"] = int(
            np.sum(valid_dt > threshold)
        )

    # Largest intervals
    n_top = min(20, len(valid_dt))

    largest = np.sort(valid_dt)[-n_top:]

    result["top20_gap_mean_s"] = float(
        np.mean(largest)
    )

    result["top20_gap_max_s"] = float(
        np.max(largest)
    )

    # Approximate missing-sample estimate
    #
    # Example:
    # expected dt = 0.005 s
    # actual dt = 0.020 s
    #
    # approximately 3 samples are missing between two
    # observed samples.
    missing_estimate = np.maximum(
        np.round(valid_dt / expected_dt) - 1,
        0
    )

    result["estimated_missing_samples"] = int(
        np.sum(missing_estimate)
    )

    # --------------------------------------------------------
    # Monotonicity
    # --------------------------------------------------------

    result["monotonic_increasing"] = bool(
        np.all(dt > 0)
    )

    # --------------------------------------------------------
    # Data-quality score
    # --------------------------------------------------------

    gap_fraction = (
        result["gaps_gt_2x"] / len(valid_dt)
    )

    jitter = result["jitter_std_percent"]

    if gap_fraction == 0 and jitter < 1:
        quality = "EXCELLENT"

    elif gap_fraction < 0.001 and jitter < 3:
        quality = "GOOD"

    elif gap_fraction < 0.01 and jitter < 10:
        quality = "MODERATE"

    else:
        quality = "POOR"

    result["quality"] = quality
    result["status"] = "OK"

    return result


# ============================================================
# SENSOR STATISTICS
# ============================================================

def calculate_sensor_statistics(df: pd.DataFrame):
    """
    Calculate basic statistics for numerical sensor columns.
    """

    stats = {}

    numeric_columns = df.select_dtypes(
        include=[np.number]
    ).columns

    for col in numeric_columns:

        if col.lower() in [
            "timestamp",
            "timestamp_sample",
            "time",
            "time_usec",
        ]:
            continue

        values = pd.to_numeric(
            df[col],
            errors="coerce"
        ).dropna()

        if len(values) < MIN_SAMPLES:
            continue

        stats[f"{col}_mean"] = safe_float(values.mean())
        stats[f"{col}_std"] = safe_float(values.std())
        stats[f"{col}_min"] = safe_float(values.min())
        stats[f"{col}_max"] = safe_float(values.max())

    return stats


# ============================================================
# FILE ANALYSIS
# ============================================================

def analyze_file(filepath: Path):
    """
    Analyze one CSV file.
    """

    print("\n" + "=" * 80)
    print(f"Analyzing: {filepath}")
    print("=" * 80)

    flight = infer_flight(filepath)
    topic = classify_topic(filepath.name)

    result = {
        "flight": flight,
        "topic": topic,
        "filename": filepath.name,
        "filepath": str(filepath),
    }

    # --------------------------------------------------------
    # Read CSV
    # --------------------------------------------------------

    try:

        df = pd.read_csv(
            filepath,
            low_memory=False
        )

    except Exception as e:

        print(f"ERROR reading file: {e}")

        result["status"] = "READ_ERROR"
        result["error"] = str(e)

        return result

    result["rows"] = int(len(df))
    result["columns"] = int(len(df.columns))

    print(f"Flight: {flight}")
    print(f"Topic:  {topic}")
    print(f"Rows:   {len(df)}")

    if len(df) < MIN_SAMPLES:

        result["status"] = "INSUFFICIENT_DATA"

        return result

    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    timestamp_column = find_timestamp_column(df)

    if timestamp_column is None:

        print("WARNING: No timestamp column found.")

        result["status"] = "NO_TIMESTAMP"

        return result

    timestamp_result = analyze_timestamps(
        df,
        timestamp_column
    )

    result.update(timestamp_result)

    # --------------------------------------------------------
    # Sensor statistics
    # --------------------------------------------------------

    sensor_stats = calculate_sensor_statistics(df)

    result.update(sensor_stats)

    # --------------------------------------------------------
    # Declared sensor sample rate
    # --------------------------------------------------------

    declared_rate_columns = [
        "sensor_sample_rate_hz",
        "sample_rate_hz",
        "sampling_rate_hz",
    ]

    for col in declared_rate_columns:

        if col in df.columns:

            values = pd.to_numeric(
                df[col],
                errors="coerce"
            ).dropna()

            if len(values) > 0:

                result["declared_rate_hz"] = float(
                    values.median()
                )

                break

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    if result.get("status") == "OK":

        print(
            f"Duration:       {result['duration_s']:.3f} s"
        )

        print(
            f"Median rate:    {result['rate_median_hz']:.3f} Hz"
        )

        print(
            f"Mean rate:      {result['rate_mean_hz']:.3f} Hz"
        )

        print(
            f"Jitter STD:     {result['jitter_std_percent']:.3f} %"
        )

        print(
            f"Jitter MAD:     {result['jitter_mad_percent']:.3f} %"
        )

        print(
            f"Duplicates:     {result['duplicate_timestamps']}"
        )

        print(
            f"Negative dt:    {result['negative_dt']}"
        )

        print(
            f"Gaps > 2x:      {result['gaps_gt_2x']}"
        )

        print(
            f"Gaps > 5x:      {result['gaps_gt_5x']}"
        )

        print(
            f"Largest gap:    {result['top20_gap_max_s']:.6f} s"
        )

        print(
            f"Quality:        {result['quality']}"
        )

    return result


# ============================================================
# FIND CSV FILES
# ============================================================

def find_candidate_files():
    """
    Find all relevant PX4 CSV files.
    """

    files = []

    for flight_dir in FLIGHT_DIRS:

        if not flight_dir.exists():

            print(
                f"WARNING: Directory does not exist:\n"
                f"  {flight_dir}"
            )

            continue

        for csv_file in sorted(
            flight_dir.glob("*.csv")
        ):

            filename = csv_file.name.lower()

            # Check include patterns
            included = any(
                pattern in filename
                for pattern in PATTERNS
            )

            if not included:
                continue

            # Check exclusions
            excluded = any(
                pattern in filename
                for pattern in EXCLUDE_PATTERNS
            )

            if excluded:
                continue

            files.append(csv_file)

    return files


# ============================================================
# RECOMMENDATION LOGIC
# ============================================================

def recommend_primary_candidates(summary_df):
    """
    Identify candidate files for primary high-rate IMU analysis.
    """

    df = summary_df.copy()

    if len(df) == 0:
        return pd.DataFrame()

    # Only valid files
    df = df[
        df["status"] == "OK"
    ].copy()

    if len(df) == 0:
        return pd.DataFrame()

    # --------------------------------------------------------
    # Topic priorities
    # --------------------------------------------------------

    topic_priority = {
        "sensor_combined": 5,
        "sensor_accel": 3,
        "sensor_gyro": 3,
        "vehicle_imu": 2,
        "sensor_gyro_fft": 1,
    }

    df["topic_priority"] = (
        df["topic"]
        .map(topic_priority)
        .fillna(0)
    )

    # --------------------------------------------------------
    # High-rate suitability
    # --------------------------------------------------------

    # Sensor_combined at approximately 200 Hz is preferred.
    df["rate_score"] = np.exp(
        -np.abs(
            np.log(
                np.maximum(
                    df["rate_median_hz"],
                    1e-9
                ) / 200.0
            )
        )
    )

    # Long datasets are preferred.
    df["duration_score"] = np.log1p(
        df["duration_s"]
    )

    # Low jitter is preferred.
    df["jitter_score"] = 1.0 / (
        1.0 +
        df["jitter_std_percent"]
    )

    # Fewer gaps are preferred.
    df["gap_score"] = 1.0 / (
        1.0 +
        df["gaps_gt_2x"]
    )

    # --------------------------------------------------------
    # Final score
    # --------------------------------------------------------

    df["candidate_score"] = (
        3.0 * df["topic_priority"]
        + 5.0 * df["rate_score"]
        + 1.0 * df["duration_score"]
        + 5.0 * df["jitter_score"]
        + 5.0 * df["gap_score"]
    )

    df = df.sort_values(
        "candidate_score",
        ascending=False
    )

    # --------------------------------------------------------
    # Candidate classification
    # --------------------------------------------------------

    def classify_candidate(row):

        topic = row["topic"]
        rate = row["rate_median_hz"]
        jitter = row["jitter_std_percent"]

        if topic == "sensor_combined":

            if 150 <= rate <= 250:

                if jitter < 5:

                    return "PRIMARY CANDIDATE"

                elif jitter < 15:

                    return "PRIMARY - INSPECT JITTER"

                else:

                    return "PRIMARY - INVESTIGATE TIMESTAMPS"

            return "SECONDARY"

        if topic in [
            "sensor_accel",
            "sensor_gyro",
        ]:

            if rate >= 50:

                return "SECONDARY HIGH-RATE"

            return "LOW-RATE - NOT PRIMARY"

        if topic == "vehicle_imu":

            return "LOW-RATE INTEGRATED IMU LOG"

        if topic == "sensor_gyro_fft":

            return "FFT AUXILIARY TOPIC"

        return "SECONDARY"

    df["recommendation"] = df.apply(
        classify_candidate,
        axis=1
    )

    return df


# ============================================================
# FLIGHT-LEVEL SUMMARY
# ============================================================

def create_flight_summary(summary_df):
    """
    Create one summary row per flight and topic.
    """

    if len(summary_df) == 0:
        return pd.DataFrame()

    valid = summary_df[
        summary_df["status"] == "OK"
    ].copy()

    if len(valid) == 0:
        return pd.DataFrame()

    group_cols = [
        "flight",
        "topic",
    ]

    grouped = (
        valid
        .groupby(group_cols)
        .agg(
            files=("filename", "count"),
            total_rows=("rows", "sum"),
            median_rate_hz=(
                "rate_median_hz",
                "median"
            ),
            median_jitter_percent=(
                "jitter_std_percent",
                "median"
            ),
            total_gaps_gt_2x=(
                "gaps_gt_2x",
                "sum"
            ),
            total_gaps_gt_5x=(
                "gaps_gt_5x",
                "sum"
            ),
            total_duplicates=(
                "duplicate_timestamps",
                "sum"
            ),
        )
        .reset_index()
    )

    return grouped


# ============================================================
# EXPORT
# ============================================================

def save_results(
    summary_df,
    candidate_df,
    flight_summary_df,
):
    """
    Save all results to CSV and Excel.
    """

    summary_csv = (
        OUTPUT_DIR /
        "sampling_rate_summary.csv"
    )

    candidate_csv = (
        OUTPUT_DIR /
        "primary_candidates.csv"
    )

    flight_csv = (
        OUTPUT_DIR /
        "flight_topic_summary.csv"
    )

    summary_df.to_csv(
        summary_csv,
        index=False
    )

    candidate_df.to_csv(
        candidate_csv,
        index=False
    )

    flight_summary_df.to_csv(
        flight_csv,
        index=False
    )

    print("\nCSV files saved:")
    print(f"  {summary_csv}")
    print(f"  {candidate_csv}")
    print(f"  {flight_csv}")

    # --------------------------------------------------------
    # Excel
    # --------------------------------------------------------

    excel_path = (
        OUTPUT_DIR /
        "sampling_rate_analysis.xlsx"
    )

    try:

        with pd.ExcelWriter(
            excel_path,
            engine="openpyxl"
        ) as writer:

            summary_df.to_excel(
                writer,
                sheet_name="File Summary",
                index=False
            )

            candidate_df.to_excel(
                writer,
                sheet_name="Candidates",
                index=False
            )

            flight_summary_df.to_excel(
                writer,
                sheet_name="Flight Summary",
                index=False
            )

        print(
            f"\nExcel file saved:\n  {excel_path}"
        )

    except ImportError:

        print(
            "\nWARNING: openpyxl is not installed."
        )

        print(
            "Install it with:"
        )

        print(
            "  python -m pip install openpyxl"
        )


# ============================================================
# PRINT FINAL REPORT
# ============================================================

def print_final_report(
    summary_df,
    candidate_df,
):
    """
    Print concise final report.
    """

    print("\n")
    print("=" * 80)
    print("FINAL SAMPLING-RATE ANALYSIS REPORT")
    print("=" * 80)

    if len(summary_df) == 0:

        print("No valid data were analyzed.")
        return

    # --------------------------------------------------------
    # By flight
    # --------------------------------------------------------

    print("\nDATA BY FLIGHT")
    print("-" * 80)

    for flight in sorted(
        summary_df["flight"].dropna().unique()
    ):

        print(f"\n{flight.upper()}")

        flight_df = summary_df[
            summary_df["flight"] == flight
        ]

        for _, row in flight_df.iterrows():

            print(
                f"  {row['topic']:20s} "
                f"{row['rate_median_hz']:10.3f} Hz | "
                f"{row['rows']:8d} samples | "
                f"jitter={row['jitter_std_percent']:.3f}% | "
                f"gaps>2x={row['gaps_gt_2x']}"
            )

    # --------------------------------------------------------
    # Candidates
    # --------------------------------------------------------

    print("\n")
    print("PRIMARY CANDIDATES")
    print("-" * 80)

    primary = candidate_df[
        candidate_df["recommendation"].str.contains(
            "PRIMARY",
            na=False
        )
    ]

    if len(primary) == 0:

        print("No primary candidate identified.")

    else:

        for _, row in primary.head(10).iterrows():

            print(
                f"{row['flight']:8s} | "
                f"{row['filename']:45s} | "
                f"{row['rate_median_hz']:.3f} Hz | "
                f"{row['rows']} samples | "
                f"jitter {row['jitter_std_percent']:.3f}% | "
                f"{row['recommendation']}"
            )

    # --------------------------------------------------------
    # Important warning
    # --------------------------------------------------------

    print("\n")
    print("SCIENTIFIC INTERPRETATION")
    print("-" * 80)

    print(
        """
1. Do NOT resample a low-rate signal to claim a higher
   information bandwidth.

2. sensor_accel / sensor_gyro files around 1-10 Hz should
   NOT be used as high-frequency vibration signals.

3. vehicle_imu is generally a low-rate log of integrated
   delta-angle / delta-velocity information and should not
   automatically be treated as the raw IMU stream.

4. sensor_combined around 200 Hz is currently the strongest
   candidate for time-domain, Allan-deviation and PSD analysis.

5. sensor_gyro_fft is an auxiliary FFT-related topic.
   Its message rate is not necessarily the raw gyro sampling
   rate. Use its internal declared sensor_sample_rate_hz only
   after validating the PX4 topic definition and data meaning.

6. Large timestamp jitter or gaps must be investigated before
   calculating Allan deviation or PSD.

7. The final scientific sampling rate must be determined from
   the actual timestamped sensor stream, not from the filename
   or the intended PX4 configuration.
"""
    )

    print("=" * 80)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("PX4 IMU SAMPLING-RATE AND DATA-QUALITY ANALYSIS")
    print("=" * 80)

    print(
        f"\nBase directory:\n{BASE_DIR}"
    )

    print(
        f"\nOutput directory:\n{OUTPUT_DIR}"
    )

    # --------------------------------------------------------
    # Find files
    # --------------------------------------------------------

    files = find_candidate_files()

    print(
        f"\nFound {len(files)} candidate CSV files."
    )

    if len(files) == 0:

        print(
            "\nERROR: No candidate CSV files found."
        )

        print(
            "Check that the following directories exist:"
        )

        for directory in FLIGHT_DIRS:
            print(f"  {directory}")

        sys.exit(1)

    # --------------------------------------------------------
    # Analyze files
    # --------------------------------------------------------

    results = []

    for filepath in files:

        result = analyze_file(filepath)

        results.append(result)

    summary_df = pd.DataFrame(results)

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    sort_columns = [
        c for c in [
            "flight",
            "topic",
            "filename"
        ]
        if c in summary_df.columns
    ]

    if sort_columns:

        summary_df = summary_df.sort_values(
            sort_columns
        )

    # --------------------------------------------------------
    # Recommendations
    # --------------------------------------------------------

    candidate_df = recommend_primary_candidates(
        summary_df
    )

    # --------------------------------------------------------
    # Flight summary
    # --------------------------------------------------------

    flight_summary_df = create_flight_summary(
        summary_df
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_results(
        summary_df,
        candidate_df,
        flight_summary_df
    )

    # --------------------------------------------------------
    # Print report
    # --------------------------------------------------------

    print_final_report(
        summary_df,
        candidate_df
    )

    print(
        "\nAnalysis completed successfully."
    )


if __name__ == "__main__":
    main()