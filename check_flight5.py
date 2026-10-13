#!/usr/bin/env python3

"""
Flight 5 IMU Data Quality Audit
--------------------------------
Purpose:
    Check whether Flight 5 is suitable to replace Flight 4
    in the IMU degradation study.

Input:
    flight5_sensor_combined_0.csv

Expected PX4 columns:
    timestamp
    accelerometer_m_s2[0]
    accelerometer_m_s2[1]
    accelerometer_m_s2[2]
    gyro_rad[0]
    gyro_rad[1]
    gyro_rad[2]

No interpolation or resampling is performed.

Outputs:
    flight5_quality_summary.csv
    flight5_window_quality.csv
    flight5_imu_statistics.csv
    flight5_quality_report.json
    figures/
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# USER SETTINGS
# ============================================================

FLIGHT_ID = 5

BASE_DIR = Path("/home/fatima/Research/validation")

INPUT_FILE = (
    BASE_DIR
    / f"flight{FLIGHT_ID}"
    / "PX4"
    / "CSV"
    / f"flight{FLIGHT_ID}_sensor_combined_0.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "imu_degradation_results"
    / f"flight{FLIGHT_ID}_quality_check"
)

FIGURE_DIR = OUTPUT_DIR / "figures"

WINDOW_SEC = 30.0

# Expected PX4 sampling rate
EXPECTED_FS = 200.0

# Timestamp quality thresholds
GOOD_JITTER_STD_PERCENT = 2.0
WARNING_JITTER_STD_PERCENT = 5.0

GOOD_GAP_RATIO = 2.0
WARNING_GAP_RATIO = 5.0

# For window quality
MAX_WINDOW_JITTER_STD_PERCENT = 2.0
MAX_WINDOW_GAP_RATIO = 2.0


# ============================================================
# COLUMN DEFINITIONS
# ============================================================

ACCEL_COLS = [
    "accelerometer_m_s2[0]",
    "accelerometer_m_s2[1]",
    "accelerometer_m_s2[2]",
]

GYRO_COLS = [
    "gyro_rad[0]",
    "gyro_rad[1]",
    "gyro_rad[2]",
]

AXES = ["x", "y", "z"]


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def mad(x):
    """Median Absolute Deviation."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return np.nan

    med = np.median(x)
    return np.median(np.abs(x - med))


def safe_rms(x):
    """Root Mean Square."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return np.nan

    return np.sqrt(np.mean(x ** 2))


def classify_quality(
    jitter_std_percent,
    gaps_gt_2x,
    gaps_gt_5x,
):
    """
    Overall timestamp quality.

    GOOD:
        low jitter and no significant gaps

    WARNING:
        moderate irregularity

    POOR:
        substantial timestamp irregularity
    """

    if (
        jitter_std_percent <= GOOD_JITTER_STD_PERCENT
        and gaps_gt_2x == 0
    ):
        return "GOOD"

    if (
        jitter_std_percent <= WARNING_JITTER_STD_PERCENT
        and gaps_gt_5x <= 1
    ):
        return "WARNING"

    return "POOR"


def calculate_timestamp_quality(timestamp_us):
    """
    Calculate sampling statistics from raw timestamps.

    No interpolation/resampling.
    """

    timestamp_us = np.asarray(timestamp_us, dtype=np.float64)

    # Remove invalid timestamps only for diagnostic calculations
    valid = np.isfinite(timestamp_us)

    timestamp_us = timestamp_us[valid]

    if len(timestamp_us) < 2:
        raise ValueError("Not enough valid timestamps.")

    # Check monotonicity
    dt_us = np.diff(timestamp_us)
    dt_s = dt_us * 1e-6

    positive_dt = dt_s[dt_s > 0]

    if len(positive_dt) == 0:
        raise ValueError("No positive timestamp differences found.")

    median_dt = np.median(positive_dt)
    mean_dt = np.mean(positive_dt)

    fs_inst = 1.0 / positive_dt

    median_fs = np.median(fs_inst)
    mean_fs = np.mean(fs_inst)

    # Jitter relative to median dt
    jitter = (positive_dt - median_dt) / median_dt * 100.0

    jitter_std_percent = np.std(jitter)
    jitter_mad_percent = np.median(np.abs(jitter - np.median(jitter)))

    gaps_gt_2x = int(np.sum(positive_dt > 2.0 * median_dt))
    gaps_gt_5x = int(np.sum(positive_dt > 5.0 * median_dt))

    largest_gap = np.max(positive_dt)

    duration = (timestamp_us[-1] - timestamp_us[0]) * 1e-6

    negative_dt_count = int(np.sum(dt_s < 0))
    zero_dt_count = int(np.sum(dt_s == 0))

    return {
        "n_timestamps": len(timestamp_us),
        "duration_s": duration,

        "median_dt_s": median_dt,
        "mean_dt_s": mean_dt,

        "median_fs_hz": median_fs,
        "mean_fs_hz": mean_fs,

        "jitter_std_percent": jitter_std_percent,
        "jitter_mad_percent": jitter_mad_percent,

        "gaps_gt_2x": gaps_gt_2x,
        "gaps_gt_5x": gaps_gt_5x,

        "largest_gap_s": largest_gap,

        "negative_dt_count": negative_dt_count,
        "zero_dt_count": zero_dt_count,

        "min_dt_s": np.min(positive_dt),
        "max_dt_s": np.max(positive_dt),
    }


def calculate_signal_statistics(df):
    """
    Global statistics for accelerometer and gyro axes.
    """

    rows = []

    signal_groups = {
        "accelerometer": ACCEL_COLS,
        "gyro": GYRO_COLS,
    }

    for sensor_type, cols in signal_groups.items():

        for axis, col in zip(AXES, cols):

            x = pd.to_numeric(df[col], errors="coerce").to_numpy()

            finite = np.isfinite(x)
            x_valid = x[finite]

            rows.append({
                "sensor": sensor_type,
                "axis": axis,
                "column": col,

                "n": len(x_valid),
                "missing_or_invalid": int(np.sum(~finite)),

                "mean": np.mean(x_valid),
                "std": np.std(x_valid, ddof=1),
                "min": np.min(x_valid),
                "max": np.max(x_valid),

                "rms": safe_rms(x_valid),
                "mad": mad(x_valid),

                "p01": np.percentile(x_valid, 1),
                "p05": np.percentile(x_valid, 5),
                "p50": np.percentile(x_valid, 50),
                "p95": np.percentile(x_valid, 95),
                "p99": np.percentile(x_valid, 99),
            })

    return pd.DataFrame(rows)


def calculate_vibration_statistics(df):
    """
    Basic vibration metrics.

    This is NOT yet the final vibration feature extraction
    used for the degradation model.
    """

    accel = df[ACCEL_COLS].to_numpy(dtype=float)
    gyro = df[GYRO_COLS].to_numpy(dtype=float)

    accel_mag = np.linalg.norm(accel, axis=1)
    gyro_mag = np.linalg.norm(gyro, axis=1)

    result = {
        "accel_magnitude_rms": safe_rms(accel_mag),
        "accel_magnitude_std": np.std(accel_mag),

        "gyro_magnitude_rms": safe_rms(gyro_mag),
        "gyro_magnitude_std": np.std(gyro_mag),

        "accel_axis_rms_mean": np.mean(
            [safe_rms(accel[:, i]) for i in range(3)]
        ),

        "gyro_axis_rms_mean": np.mean(
            [safe_rms(gyro[:, i]) for i in range(3)]
        ),
    }

    return result


def calculate_window_quality(df):
    """
    30-second window quality.

    Used to determine whether Flight 5 contains
    continuous near-uniform sections suitable for
    later Welch PSD / Allan analysis.
    """

    timestamp_us = df["timestamp"].to_numpy(dtype=np.float64)

    start = timestamp_us[0]
    end = timestamp_us[-1]

    window_edges = np.arange(
        start,
        end + WINDOW_SEC * 1e6,
        WINDOW_SEC * 1e6
    )

    rows = []

    for i in range(len(window_edges) - 1):

        t0 = window_edges[i]
        t1 = window_edges[i + 1]

        mask = (
            (timestamp_us >= t0)
            & (timestamp_us < t1)
        )

        idx = np.where(mask)[0]

        if len(idx) < 100:
            continue

        ts = timestamp_us[idx]

        dt_s = np.diff(ts) * 1e-6
        positive_dt = dt_s[dt_s > 0]

        if len(positive_dt) < 2:
            continue

        median_dt = np.median(positive_dt)

        fs = 1.0 / positive_dt

        jitter = (
            (positive_dt - median_dt)
            / median_dt
            * 100.0
        )

        gaps_gt_2x = int(
            np.sum(positive_dt > 2.0 * median_dt)
        )

        gaps_gt_5x = int(
            np.sum(positive_dt > 5.0 * median_dt)
        )

        jitter_std = np.std(jitter)

        duration = (
            (ts[-1] - ts[0]) * 1e-6
        )

        if (
            jitter_std <= MAX_WINDOW_JITTER_STD_PERCENT
            and gaps_gt_2x == 0
        ):
            quality = "GOOD"

        elif (
            jitter_std <= 5.0
            and gaps_gt_5x <= 1
        ):
            quality = "WARNING"

        else:
            quality = "POOR"

        rows.append({
            "window": i,
            "start_s": (t0 - start) * 1e-6,
            "end_s": (t1 - start) * 1e-6,

            "n_samples": len(idx),
            "duration_s": duration,

            "median_dt_s": np.median(positive_dt),
            "mean_dt_s": np.mean(positive_dt),

            "median_fs_hz": np.median(fs),
            "mean_fs_hz": np.mean(fs),

            "jitter_std_percent": jitter_std,

            "gaps_gt_2x": gaps_gt_2x,
            "gaps_gt_5x": gaps_gt_5x,

            "largest_gap_s": np.max(positive_dt),

            "quality": quality,
        })

    return pd.DataFrame(rows)


# ============================================================
# PLOTS
# ============================================================

def make_plots(df, window_df, timestamp_quality):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = df["timestamp"].to_numpy(dtype=float)
    time_s = (timestamp - timestamp[0]) * 1e-6

    # --------------------------------------------------------
    # 1. Accelerometer
    # --------------------------------------------------------

    fig = plt.figure(figsize=(14, 8))

    for i, col in enumerate(ACCEL_COLS):

        plt.plot(
            time_s,
            df[col].to_numpy(),
            label=f"Accel {AXES[i]}"
        )

    plt.xlabel("Time [s]")
    plt.ylabel("Acceleration [m/s²]")
    plt.title(f"Flight {FLIGHT_ID} - Accelerometer")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / "flight5_accelerometer.png",
        dpi=200
    )
    plt.close()

    # --------------------------------------------------------
    # 2. Gyroscope
    # --------------------------------------------------------

    fig = plt.figure(figsize=(14, 8))

    for i, col in enumerate(GYRO_COLS):

        plt.plot(
            time_s,
            df[col].to_numpy(),
            label=f"Gyro {AXES[i]}"
        )

    plt.xlabel("Time [s]")
    plt.ylabel("Angular rate [rad/s]")
    plt.title(f"Flight {FLIGHT_ID} - Gyroscope")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / "flight5_gyroscope.png",
        dpi=200
    )
    plt.close()

    # --------------------------------------------------------
    # 3. Sampling interval
    # --------------------------------------------------------

    dt_s = np.diff(timestamp) * 1e-6

    fig = plt.figure(figsize=(14, 6))

    plt.plot(
        time_s[1:],
        dt_s * 1000,
        linewidth=0.8
    )

    plt.axhline(
        timestamp_quality["median_dt_s"] * 1000,
        linestyle="--",
        label="Median dt"
    )

    plt.xlabel("Time [s]")
    plt.ylabel("dt [ms]")
    plt.title(f"Flight {FLIGHT_ID} - Timestamp Interval")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / "flight5_dt.png",
        dpi=200
    )
    plt.close()

    # --------------------------------------------------------
    # 4. Instantaneous sampling frequency
    # --------------------------------------------------------

    positive = dt_s > 0

    fs = np.full_like(dt_s, np.nan)
    fs[positive] = 1.0 / dt_s[positive]

    fig = plt.figure(figsize=(14, 6))

    plt.plot(
        time_s[1:],
        fs,
        linewidth=0.8
    )

    plt.axhline(
        EXPECTED_FS,
        linestyle="--",
        label="Expected 200 Hz"
    )

    plt.xlabel("Time [s]")
    plt.ylabel("Sampling frequency [Hz]")
    plt.title(f"Flight {FLIGHT_ID} - Instantaneous Sampling Frequency")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / "flight5_sampling_frequency.png",
        dpi=200
    )
    plt.close()

    # --------------------------------------------------------
    # 5. Window quality
    # --------------------------------------------------------

    if len(window_df) > 0:

        fig = plt.figure(figsize=(14, 6))

        x = window_df["start_s"]

        plt.plot(
            x,
            window_df["median_fs_hz"],
            marker="o",
            label="Median Fs"
        )

        plt.axhline(
            EXPECTED_FS,
            linestyle="--",
            label="Expected 200 Hz"
        )

        plt.xlabel("Window start [s]")
        plt.ylabel("Frequency [Hz]")
        plt.title(
            f"Flight {FLIGHT_ID} - 30 s Window Sampling Quality"
        )

        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.tight_layout()

        plt.savefig(
            FIGURE_DIR / "flight5_window_sampling.png",
            dpi=200
        )

        plt.close()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(f"FLIGHT {FLIGHT_ID} - IMU QUALITY AUDIT")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Check input
    # --------------------------------------------------------

    print("\nInput:")
    print(INPUT_FILE)

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"\nFlight {FLIGHT_ID} file was not found:\n"
            f"{INPUT_FILE}\n\n"
            "Please check the flight5 folder and filename."
        )

    # --------------------------------------------------------
    # Load CSV
    # --------------------------------------------------------

    print("\nLoading CSV...")

    df = pd.read_csv(INPUT_FILE)

    print(f"Rows    : {len(df):,}")
    print(f"Columns : {len(df.columns)}")

    # --------------------------------------------------------
    # Column validation
    # --------------------------------------------------------

    required_cols = (
        ["timestamp"]
        + ACCEL_COLS
        + GYRO_COLS
    )

    missing_cols = [
        col
        for col in required_cols
        if col not in df.columns
    ]

    if missing_cols:

        print("\nERROR: Missing required columns:")

        for col in missing_cols:
            print("  -", col)

        print("\nAvailable columns:")
        for col in df.columns:
            print("  -", col)

        raise ValueError(
            "Required PX4 sensor_combined columns are missing."
        )

    print("\nRequired columns found: OK")

    # --------------------------------------------------------
    # Convert numeric
    # --------------------------------------------------------

    for col in required_cols:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Timestamp validation
    # --------------------------------------------------------

    timestamp = df["timestamp"].to_numpy(dtype=float)

    finite_timestamp = np.isfinite(timestamp)

    invalid_timestamp_count = int(
        np.sum(~finite_timestamp)
    )

    print(
        f"\nInvalid timestamps: "
        f"{invalid_timestamp_count:,}"
    )

    if invalid_timestamp_count > 0:

        df = df.loc[
            finite_timestamp
        ].reset_index(drop=True)

        timestamp = df["timestamp"].to_numpy(dtype=float)

    timestamp_quality = calculate_timestamp_quality(
        timestamp
    )

    timestamp_quality["invalid_timestamp_count"] = (
        invalid_timestamp_count
    )

    timestamp_quality["quality"] = classify_quality(
        timestamp_quality["jitter_std_percent"],
        timestamp_quality["gaps_gt_2x"],
        timestamp_quality["gaps_gt_5x"],
    )

    # --------------------------------------------------------
    # Print timestamp report
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("TIMESTAMP QUALITY")
    print("-" * 70)

    print(
        f"Duration              : "
        f"{timestamp_quality['duration_s']:.3f} s"
    )

    print(
        f"Median dt             : "
        f"{timestamp_quality['median_dt_s']:.6f} s"
    )

    print(
        f"Mean dt               : "
        f"{timestamp_quality['mean_dt_s']:.6f} s"
    )

    print(
        f"Median Fs             : "
        f"{timestamp_quality['median_fs_hz']:.3f} Hz"
    )

    print(
        f"Mean Fs               : "
        f"{timestamp_quality['mean_fs_hz']:.3f} Hz"
    )

    print(
        f"Jitter STD            : "
        f"{timestamp_quality['jitter_std_percent']:.6f} %"
    )

    print(
        f"Jitter MAD            : "
        f"{timestamp_quality['jitter_mad_percent']:.6f} %"
    )

    print(
        f"Gaps > 2x median dt   : "
        f"{timestamp_quality['gaps_gt_2x']:,}"
    )

    print(
        f"Gaps > 5x median dt   : "
        f"{timestamp_quality['gaps_gt_5x']:,}"
    )

    print(
        f"Largest gap           : "
        f"{timestamp_quality['largest_gap_s']:.6f} s"
    )

    print(
        f"Negative dt           : "
        f"{timestamp_quality['negative_dt_count']:,}"
    )

    print(
        f"Zero dt               : "
        f"{timestamp_quality['zero_dt_count']:,}"
    )

    print(
        f"Timestamp quality     : "
        f"{timestamp_quality['quality']}"
    )

    # --------------------------------------------------------
    # Signal statistics
    # --------------------------------------------------------

    print("\nCalculating signal statistics...")

    signal_stats = calculate_signal_statistics(df)

    print("\n" + "-" * 70)
    print("GLOBAL IMU STATISTICS")
    print("-" * 70)

    print(
        signal_stats[
            [
                "sensor",
                "axis",
                "mean",
                "std",
                "rms",
                "mad",
                "min",
                "max",
            ]
        ].to_string(index=False)
    )

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    missing_rows = []

    for col in required_cols:

        count = int(
            df[col].isna().sum()
        )

        missing_rows.append({
            "column": col,
            "missing_count": count,
            "missing_percent":
                100.0 * count / len(df),
        })

    missing_df = pd.DataFrame(missing_rows)

    # --------------------------------------------------------
    # Vibration statistics
    # --------------------------------------------------------

    vibration_stats = calculate_vibration_statistics(df)

    print("\n" + "-" * 70)
    print("GLOBAL VIBRATION SUMMARY")
    print("-" * 70)

    for key, value in vibration_stats.items():

        print(
            f"{key:35s}: "
            f"{value:.6f}"
        )

    # --------------------------------------------------------
    # Window quality
    # --------------------------------------------------------

    print("\nCalculating 30-second window quality...")

    window_df = calculate_window_quality(df)

    print(
        f"Number of 30-s windows: "
        f"{len(window_df)}"
    )

    if len(window_df) > 0:

        quality_counts = (
            window_df["quality"]
            .value_counts()
            .to_dict()
        )

        print(
            "Window quality:",
            quality_counts
        )

    # --------------------------------------------------------
    # Clipping columns if present
    # --------------------------------------------------------

    clipping_summary = {}

    clipping_columns = [
        "accelerometer_clipping",
        "gyro_clipping",
    ]

    for col in clipping_columns:

        if col in df.columns:

            values = pd.to_numeric(
                df[col],
                errors="coerce"
            )

            clipping_summary[col] = {
                "nonzero_count": int(
                    np.sum(values.fillna(0) != 0)
                ),
                "max_value": (
                    float(values.max())
                    if values.notna().any()
                    else np.nan
                ),
            }

    # --------------------------------------------------------
    # Sampling-rate comparison
    # --------------------------------------------------------

    fs_error_percent = (
        abs(
            timestamp_quality["median_fs_hz"]
            - EXPECTED_FS
        )
        / EXPECTED_FS
        * 100.0
    )

    timestamp_quality[
        "median_fs_error_percent"
    ] = fs_error_percent

    # --------------------------------------------------------
    # Suitability decision
    # --------------------------------------------------------

    # This is a technical data-quality classification,
    # not a scientific degradation conclusion.

    if (
        timestamp_quality["quality"] == "GOOD"
        and fs_error_percent <= 2.0
        and len(window_df) > 0
    ):

        suitable_for_main_analysis = True

    elif (
        timestamp_quality["quality"] == "WARNING"
        and fs_error_percent <= 5.0
        and len(window_df) > 0
    ):

        suitable_for_main_analysis = "CONDITIONAL"

    else:

        suitable_for_main_analysis = False

    timestamp_quality[
        "suitable_for_main_analysis"
    ] = suitable_for_main_analysis

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    print("\nGenerating figures...")

    make_plots(
        df,
        window_df,
        timestamp_quality
    )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    summary_rows = []

    for key, value in timestamp_quality.items():

        summary_rows.append({
            "metric": key,
            "value": value
        })

    summary_df = pd.DataFrame(summary_rows)

    summary_df.to_csv(
        OUTPUT_DIR / "flight5_quality_summary.csv",
        index=False
    )

    signal_stats.to_csv(
        OUTPUT_DIR / "flight5_imu_statistics.csv",
        index=False
    )

    window_df.to_csv(
        OUTPUT_DIR / "flight5_window_quality.csv",
        index=False
    )

    missing_df.to_csv(
        OUTPUT_DIR / "flight5_missing_values.csv",
        index=False
    )

    # --------------------------------------------------------
    # Excel
    # --------------------------------------------------------

    excel_file = (
        OUTPUT_DIR
        / "flight5_quality_report.xlsx"
    )

    with pd.ExcelWriter(
        excel_file,
        engine="openpyxl"
    ) as writer:

        summary_df.to_excel(
            writer,
            sheet_name="Summary",
            index=False
        )

        signal_stats.to_excel(
            writer,
            sheet_name="IMU_Statistics",
            index=False
        )

        window_df.to_excel(
            writer,
            sheet_name="Window_Quality",
            index=False
        )

        missing_df.to_excel(
            writer,
            sheet_name="Missing_Values",
            index=False
        )

        pd.DataFrame(
            [vibration_stats]
        ).to_excel(
            writer,
            sheet_name="Vibration",
            index=False
        )

        if clipping_summary:

            clipping_df = pd.DataFrame(
                clipping_summary
            ).T.reset_index()

            clipping_df.columns = [
                "column",
                "nonzero_count",
                "max_value"
            ]

            clipping_df.to_excel(
                writer,
                sheet_name="Clipping",
                index=False
            )

    # --------------------------------------------------------
    # JSON report
    # --------------------------------------------------------

    report = {
        "flight": FLIGHT_ID,
        "input_file": str(INPUT_FILE),
        "output_directory": str(OUTPUT_DIR),

        "expected_sampling_frequency_hz":
            EXPECTED_FS,

        "window_length_s":
            WINDOW_SEC,

        "timestamp_quality":
            timestamp_quality,

        "vibration_statistics":
            vibration_stats,

        "clipping":
            clipping_summary,

        "n_rows":
            int(len(df)),

        "n_columns":
            int(len(df.columns)),
    }

    with open(
        OUTPUT_DIR / "flight5_quality_report.json",
        "w"
    ) as f:

        json.dump(
            report,
            f,
            indent=4,
            default=str
        )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL FLIGHT 5 ASSESSMENT")
    print("=" * 70)

    print(
        f"Timestamp quality      : "
        f"{timestamp_quality['quality']}"
    )

    print(
        f"Median sampling rate   : "
        f"{timestamp_quality['median_fs_hz']:.3f} Hz"
    )

    print(
        f"Jitter STD             : "
        f"{timestamp_quality['jitter_std_percent']:.3f} %"
    )

    print(
        f"Gaps > 2x              : "
        f"{timestamp_quality['gaps_gt_2x']:,}"
    )

    print(
        f"Gaps > 5x              : "
        f"{timestamp_quality['gaps_gt_5x']:,}"
    )

    print(
        f"30-s windows           : "
        f"{len(window_df)}"
    )

    print(
        f"Main-analysis status   : "
        f"{suitable_for_main_analysis}"
    )

    print("\nOutput:")
    print(OUTPUT_DIR)

    print("\nFigures:")
    print(FIGURE_DIR)

    print("\nDone.")


if __name__ == "__main__":
    main()