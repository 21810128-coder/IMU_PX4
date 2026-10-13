#!/usr/bin/env python3

"""
================================================================================
MULTI-FLIGHT IMU DATA QUALITY AUDIT
================================================================================
Purpose:
    Check whether each flight is suitable for IMU degradation analysis.
    Supports multiple flights: F2, F3, F5, F6, F7

Input:
    flightX_sensor_combined_0.csv for each flight

Expected PX4 columns:
    timestamp
    accelerometer_m_s2[0], [1], [2]
    gyro_rad[0], [1], [2]

No interpolation or resampling is performed.

Outputs:
    Per-flight quality reports
    Cross-flight comparison table
    Figures
================================================================================
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


# ============================================================
# CONFIGURATION
# ============================================================

# All flights to analyze
FLIGHTS = {
    "flight2": {
        "id": 2,
        "path": Path("/home/fatima/Research/validation/flight2/PX4/CSV/flight2_sensor_combined_0.csv"),
        "role": "baseline",
    },
    "flight3": {
        "id": 3,
        "path": Path("/home/fatima/Research/validation/flight3/PX4/CSV/flight3_sensor_combined_0.csv"),
        "role": "long_duration",
    },
    "flight5": {
        "id": 5,
        "path": Path("/home/fatima/Research/validation/flight5/PX4/CSV/flight5_sensor_combined_0.csv"),
        "role": "validation",
    },
    "flight6": {
        "id": 6,
        "path": Path("/home/fatima/Research/validation/flight6/PX4/CSV/flight6_sensor_combined_0.csv"),
        "role": "new_flight_fixed_wing",
    },
    "flight7": {
        "id": 7,
        "path": Path("/home/fatima/Research/validation/flight7/PX4/CSV/flight7_sensor_combined_0.csv"),
        "role": "new_flight_fixed_wing",
    },
}

BASE_OUTPUT_DIR = Path(
    "/home/fatima/Research/validation/imu_degradation_results/quality_audit"
)

# Window parameters
WINDOW_SEC = 30.0
EXPECTED_FS = 200.0

# Quality thresholds
GOOD_JITTER_STD_PERCENT = 2.0
WARNING_JITTER_STD_PERCENT = 5.0
GOOD_GAP_RATIO = 2.0
WARNING_GAP_RATIO = 5.0
MAX_WINDOW_JITTER_STD_PERCENT = 2.0
MAX_WINDOW_GAP_RATIO = 2.0


# Column definitions
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


def classify_quality(jitter_std_percent, gaps_gt_2x, gaps_gt_5x):
    """Overall timestamp quality."""
    if jitter_std_percent <= GOOD_JITTER_STD_PERCENT and gaps_gt_2x == 0:
        return "GOOD"
    if jitter_std_percent <= WARNING_JITTER_STD_PERCENT and gaps_gt_5x <= 1:
        return "WARNING"
    return "POOR"


def calculate_timestamp_quality(timestamp_us):
    """Calculate sampling statistics from raw timestamps."""
    timestamp_us = np.asarray(timestamp_us, dtype=np.float64)
    valid = np.isfinite(timestamp_us)
    timestamp_us = timestamp_us[valid]
    
    if len(timestamp_us) < 2:
        raise ValueError("Not enough valid timestamps.")
    
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
    """Global statistics for accelerometer and gyro axes."""
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
            
            if len(x_valid) == 0:
                continue
            
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
    """Basic vibration metrics."""
    accel = df[ACCEL_COLS].to_numpy(dtype=float)
    gyro = df[GYRO_COLS].to_numpy(dtype=float)
    
    accel_mag = np.linalg.norm(accel, axis=1)
    gyro_mag = np.linalg.norm(gyro, axis=1)
    
    result = {
        "accel_magnitude_rms": safe_rms(accel_mag),
        "accel_magnitude_std": np.std(accel_mag),
        "gyro_magnitude_rms": safe_rms(gyro_mag),
        "gyro_magnitude_std": np.std(gyro_mag),
        "accel_axis_rms_mean": np.mean([safe_rms(accel[:, i]) for i in range(3)]),
        "gyro_axis_rms_mean": np.mean([safe_rms(gyro[:, i]) for i in range(3)]),
    }
    
    return result


def calculate_window_quality(df, window_sec=30.0):
    """30-second window quality."""
    timestamp_us = df["timestamp"].to_numpy(dtype=np.float64)
    start = timestamp_us[0]
    end = timestamp_us[-1]
    
    window_edges = np.arange(start, end + window_sec * 1e6, window_sec * 1e6)
    rows = []
    
    for i in range(len(window_edges) - 1):
        t0 = window_edges[i]
        t1 = window_edges[i + 1]
        mask = (timestamp_us >= t0) & (timestamp_us < t1)
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
        jitter = (positive_dt - median_dt) / median_dt * 100.0
        gaps_gt_2x = int(np.sum(positive_dt > 2.0 * median_dt))
        gaps_gt_5x = int(np.sum(positive_dt > 5.0 * median_dt))
        jitter_std = np.std(jitter)
        duration = (ts[-1] - ts[0]) * 1e-6
        
        if jitter_std <= MAX_WINDOW_JITTER_STD_PERCENT and gaps_gt_2x == 0:
            quality = "GOOD"
        elif jitter_std <= 5.0 and gaps_gt_5x <= 1:
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

def make_plots(df, window_df, timestamp_quality, flight_id, figure_dir):
    """Generate quality plots for a single flight."""
    figure_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = df["timestamp"].to_numpy(dtype=float)
    time_s = (timestamp - timestamp[0]) * 1e-6
    
    # 1. Accelerometer
    fig = plt.figure(figsize=(14, 8))
    for i, col in enumerate(ACCEL_COLS):
        plt.plot(time_s, df[col].to_numpy(), label=f"Accel {AXES[i]}", linewidth=0.5)
    plt.xlabel("Time [s]")
    plt.ylabel("Acceleration [m/s²]")
    plt.title(f"Flight {flight_id} - Accelerometer")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_dir / f"flight{flight_id}_accelerometer.png", dpi=150)
    plt.close()
    
    # 2. Gyroscope
    fig = plt.figure(figsize=(14, 8))
    for i, col in enumerate(GYRO_COLS):
        plt.plot(time_s, df[col].to_numpy(), label=f"Gyro {AXES[i]}", linewidth=0.5)
    plt.xlabel("Time [s]")
    plt.ylabel("Angular rate [rad/s]")
    plt.title(f"Flight {flight_id} - Gyroscope")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_dir / f"flight{flight_id}_gyroscope.png", dpi=150)
    plt.close()
    
    # 3. Sampling interval
    dt_s = np.diff(timestamp) * 1e-6
    fig = plt.figure(figsize=(14, 6))
    plt.plot(time_s[1:], dt_s * 1000, linewidth=0.5)
    plt.axhline(timestamp_quality["median_dt_s"] * 1000, linestyle="--", 
                color='red', label="Median dt")
    plt.xlabel("Time [s]")
    plt.ylabel("dt [ms]")
    plt.title(f"Flight {flight_id} - Timestamp Interval")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_dir / f"flight{flight_id}_dt.png", dpi=150)
    plt.close()
    
    # 4. Window quality
    if len(window_df) > 0:
        fig = plt.figure(figsize=(14, 6))
        x = window_df["start_s"]
        plt.plot(x, window_df["median_fs_hz"], marker="o", linestyle='-', 
                 markersize=3, label="Median Fs")
        plt.axhline(EXPECTED_FS, linestyle="--", color='red', label="Expected 200 Hz")
        plt.xlabel("Window start [s]")
        plt.ylabel("Frequency [Hz]")
        plt.title(f"Flight {flight_id} - 30s Window Sampling Quality")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(figure_dir / f"flight{flight_id}_window_sampling.png", dpi=150)
        plt.close()


# ============================================================
# MAIN FLIGHT PROCESSING
# ============================================================

def process_flight(flight_name, flight_config):
    """Process a single flight."""
    flight_id = flight_config["id"]
    input_file = flight_config["path"]
    role = flight_config.get("role", "unknown")
    
    print("\n" + "=" * 70)
    print(f"FLIGHT {flight_id} - IMU QUALITY AUDIT")
    print(f"Role: {role}")
    print("=" * 70)
    
    output_dir = BASE_OUTPUT_DIR / f"flight{flight_id}_quality"
    figure_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nInput: {input_file}")
    
    if not input_file.exists():
        print(f"❌ File not found: {input_file}")
        return None
    
    # Load CSV
    print("\nLoading CSV...")
    df = pd.read_csv(input_file)
    
    print(f"Rows    : {len(df):,}")
    print(f"Columns : {len(df.columns)}")
    
    # Validate columns
    required_cols = ["timestamp"] + ACCEL_COLS + GYRO_COLS
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"\n❌ Missing columns: {missing_cols}")
        print(f"Available columns: {df.columns.tolist()}")
        return None
    
    print("✅ Required columns found")
    
    # Convert to numeric
    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Timestamp validation
    timestamp = df["timestamp"].to_numpy(dtype=float)
    finite_timestamp = np.isfinite(timestamp)
    invalid_timestamp_count = int(np.sum(~finite_timestamp))
    
    print(f"\nInvalid timestamps: {invalid_timestamp_count:,}")
    
    if invalid_timestamp_count > 0:
        df = df.loc[finite_timestamp].reset_index(drop=True)
        timestamp = df["timestamp"].to_numpy(dtype=float)
    
    # Timestamp quality
    timestamp_quality = calculate_timestamp_quality(timestamp)
    timestamp_quality["invalid_timestamp_count"] = invalid_timestamp_count
    timestamp_quality["quality"] = classify_quality(
        timestamp_quality["jitter_std_percent"],
        timestamp_quality["gaps_gt_2x"],
        timestamp_quality["gaps_gt_5x"],
    )
    
    # Print timestamp report
    print("\n" + "-" * 70)
    print("TIMESTAMP QUALITY")
    print("-" * 70)
    print(f"Duration              : {timestamp_quality['duration_s']:.3f} s")
    print(f"Median dt             : {timestamp_quality['median_dt_s']*1000:.3f} ms")
    print(f"Mean dt               : {timestamp_quality['mean_dt_s']*1000:.3f} ms")
    print(f"Median Fs             : {timestamp_quality['median_fs_hz']:.3f} Hz")
    print(f"Mean Fs               : {timestamp_quality['mean_fs_hz']:.3f} Hz")
    print(f"Jitter STD            : {timestamp_quality['jitter_std_percent']:.6f} %")
    print(f"Jitter MAD            : {timestamp_quality['jitter_mad_percent']:.6f} %")
    print(f"Gaps > 2x median dt   : {timestamp_quality['gaps_gt_2x']:,}")
    print(f"Gaps > 5x median dt   : {timestamp_quality['gaps_gt_5x']:,}")
    print(f"Largest gap           : {timestamp_quality['largest_gap_s']*1000:.3f} ms")
    print(f"Negative dt           : {timestamp_quality['negative_dt_count']:,}")
    print(f"Zero dt               : {timestamp_quality['zero_dt_count']:,}")
    print(f"Timestamp quality     : {timestamp_quality['quality']}")
    
    # Signal statistics
    print("\nCalculating signal statistics...")
    signal_stats = calculate_signal_statistics(df)
    
    print("\n" + "-" * 70)
    print("GLOBAL IMU STATISTICS")
    print("-" * 70)
    print(signal_stats[["sensor", "axis", "mean", "std", "rms", "mad"]].to_string(index=False))
    
    # Vibration statistics
    vibration_stats = calculate_vibration_statistics(df)
    
    print("\n" + "-" * 70)
    print("GLOBAL VIBRATION SUMMARY")
    print("-" * 70)
    for key, value in vibration_stats.items():
        print(f"{key:35s}: {value:.6f}")
    
    # Window quality
    print("\nCalculating 30-second window quality...")
    window_df = calculate_window_quality(df, WINDOW_SEC)
    
    print(f"Number of 30-s windows: {len(window_df)}")
    
    if len(window_df) > 0:
        quality_counts = window_df["quality"].value_counts().to_dict()
        print(f"Window quality: {quality_counts}")
    
    # Clipping columns
    clipping_summary = {}
    clipping_columns = ["accelerometer_clipping", "gyro_clipping"]
    
    for col in clipping_columns:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce")
            clipping_summary[col] = {
                "nonzero_count": int(np.sum(values.fillna(0) != 0)),
                "max_value": float(values.max()) if values.notna().any() else np.nan,
            }
    
    # Suitability decision
    fs_error_percent = abs(timestamp_quality["median_fs_hz"] - EXPECTED_FS) / EXPECTED_FS * 100.0
    timestamp_quality["median_fs_error_percent"] = fs_error_percent
    
    if (timestamp_quality["quality"] == "GOOD" and 
        fs_error_percent <= 2.0 and 
        len(window_df) > 0):
        suitable = True
    elif (timestamp_quality["quality"] == "WARNING" and 
          fs_error_percent <= 5.0 and 
          len(window_df) > 0):
        suitable = "CONDITIONAL"
    else:
        suitable = False
    
    timestamp_quality["suitable_for_main_analysis"] = suitable
    
    # Plots
    print("\nGenerating figures...")
    make_plots(df, window_df, timestamp_quality, flight_id, figure_dir)
    
    # Save CSVs
    summary_rows = [{"metric": k, "value": v} for k, v in timestamp_quality.items()]
    summary_df = pd.DataFrame(summary_rows)
    
    summary_df.to_csv(output_dir / f"flight{flight_id}_quality_summary.csv", index=False)
    signal_stats.to_csv(output_dir / f"flight{flight_id}_imu_statistics.csv", index=False)
    window_df.to_csv(output_dir / f"flight{flight_id}_window_quality.csv", index=False)
    
    # Excel report
    excel_file = output_dir / f"flight{flight_id}_quality_report.xlsx"
    with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        signal_stats.to_excel(writer, sheet_name="IMU_Statistics", index=False)
        window_df.to_excel(writer, sheet_name="Window_Quality", index=False)
        pd.DataFrame([vibration_stats]).to_excel(writer, sheet_name="Vibration", index=False)
    
    # JSON report
    report = {
        "flight": flight_id,
        "role": role,
        "input_file": str(input_file),
        "expected_sampling_frequency_hz": EXPECTED_FS,
        "window_length_s": WINDOW_SEC,
        "timestamp_quality": timestamp_quality,
        "vibration_statistics": vibration_stats,
        "clipping": clipping_summary,
        "n_rows": int(len(df)),
        "n_columns": int(len(df.columns)),
    }
    
    with open(output_dir / f"flight{flight_id}_quality_report.json", "w") as f:
        json.dump(report, f, indent=4, default=str)
    
    # Final report
    print("\n" + "=" * 70)
    print(f"FINAL FLIGHT {flight_id} ASSESSMENT")
    print("=" * 70)
    print(f"Timestamp quality      : {timestamp_quality['quality']}")
    print(f"Median sampling rate   : {timestamp_quality['median_fs_hz']:.3f} Hz")
    print(f"Jitter STD             : {timestamp_quality['jitter_std_percent']:.3f} %")
    print(f"Gaps > 2x              : {timestamp_quality['gaps_gt_2x']:,}")
    print(f"Gaps > 5x              : {timestamp_quality['gaps_gt_5x']:,}")
    print(f"30-s windows           : {len(window_df)}")
    print(f"Main-analysis status   : {suitable}")
    
    return {
        "flight": flight_id,
        "role": role,
        "n_rows": len(df),
        "duration_s": timestamp_quality["duration_s"],
        "median_fs_hz": timestamp_quality["median_fs_hz"],
        "jitter_std_percent": timestamp_quality["jitter_std_percent"],
        "gaps_gt_2x": timestamp_quality["gaps_gt_2x"],
        "gaps_gt_5x": timestamp_quality["gaps_gt_5x"],
        "n_windows": len(window_df),
        "quality": timestamp_quality["quality"],
        "suitable": suitable,
        "accel_magnitude_rms": vibration_stats["accel_magnitude_rms"],
        "gyro_magnitude_rms": vibration_stats["gyro_magnitude_rms"],
        "accel_magnitude_std": vibration_stats["accel_magnitude_std"],
        "gyro_magnitude_std": vibration_stats["gyro_magnitude_std"],
    }


# ============================================================
# CROSS-FLIGHT COMPARISON
# ============================================================

def create_comparison_table(all_results):
    """Create comparison table across all flights."""
    print("\n" + "=" * 70)
    print("CROSS-FLIGHT COMPARISON")
    print("=" * 70)
    
    df = pd.DataFrame(all_results)
    
    # Save
    comparison_path = BASE_OUTPUT_DIR / "all_flights_comparison.csv"
    df.to_csv(comparison_path, index=False)
    print(f"\n✅ Comparison saved: {comparison_path}")
    
    print("\n" + df.to_string(index=False))
    
    return df


def plot_cross_flight_comparison(comparison_df):
    """Plot cross-flight comparison."""
    figure_dir = BASE_OUTPUT_DIR / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    flights = comparison_df["flight"].astype(str).tolist()
    x = np.arange(len(flights))
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
    
    # 1. Duration
    ax = axes[0, 0]
    ax.bar(x, comparison_df["duration_s"], color=colors[:len(flights)], 
           edgecolor='black', alpha=0.8)
    ax.set_xlabel("Flight")
    ax.set_ylabel("Duration (s)")
    ax.set_title("(a) Flight Duration")
    ax.set_xticks(x)
    ax.set_xticklabels(flights)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 2. Median Fs
    ax = axes[0, 1]
    ax.bar(x, comparison_df["median_fs_hz"], color=colors[:len(flights)], 
           edgecolor='black', alpha=0.8)
    ax.axhline(200, linestyle='--', color='red', label='Expected 200 Hz')
    ax.set_xlabel("Flight")
    ax.set_ylabel("Median Fs (Hz)")
    ax.set_title("(b) Median Sampling Frequency")
    ax.set_xticks(x)
    ax.set_xticklabels(flights)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Jitter
    ax = axes[0, 2]
    ax.bar(x, comparison_df["jitter_std_percent"], color=colors[:len(flights)], 
           edgecolor='black', alpha=0.8)
    ax.axhline(2.0, linestyle='--', color='green', label='GOOD threshold')
    ax.axhline(5.0, linestyle='--', color='orange', label='WARNING threshold')
    ax.set_xlabel("Flight")
    ax.set_ylabel("Jitter STD (%)")
    ax.set_title("(c) Timestamp Jitter")
    ax.set_xticks(x)
    ax.set_xticklabels(flights)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. Gaps
    ax = axes[1, 0]
    width = 0.35
    ax.bar(x - width/2, comparison_df["gaps_gt_2x"], width, 
           label='>2×dt', color='orange', edgecolor='black', alpha=0.8)
    ax.bar(x + width/2, comparison_df["gaps_gt_5x"], width, 
           label='>5×dt', color='red', edgecolor='black', alpha=0.8)
    ax.set_xlabel("Flight")
    ax.set_ylabel("Number of Gaps")
    ax.set_title("(d) Timestamp Gaps")
    ax.set_xticks(x)
    ax.set_xticklabels(flights)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 5. Number of Windows
    ax = axes[1, 1]
    ax.bar(x, comparison_df["n_windows"], color=colors[:len(flights)], 
           edgecolor='black', alpha=0.8)
    ax.set_xlabel("Flight")
    ax.set_ylabel("Number of 30-s Windows")
    ax.set_title("(e) Analysis Windows")
    ax.set_xticks(x)
    ax.set_xticklabels(flights)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 6. Vibration (Accel Magnitude RMS)
    ax = axes[1, 2]
    ax.bar(x, comparison_df["accel_magnitude_rms"], color=colors[:len(flights)], 
           edgecolor='black', alpha=0.8)
    ax.set_xlabel("Flight")
    ax.set_ylabel("Accel Magnitude RMS (m/s²)")
    ax.set_title("(f) Global Vibration Level")
    ax.set_xticks(x)
    ax.set_xticklabels(flights)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle("Cross-Flight Quality Comparison", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = figure_dir / "cross_flight_comparison.png"
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Comparison figure saved: {save_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    """Main execution."""
    print("=" * 70)
    print("MULTI-FLIGHT IMU QUALITY AUDIT")
    print("=" * 70)
    
    print("\nFlights to analyze:")
    for flight_name, config in FLIGHTS.items():
        print(f"  {flight_name} (ID={config['id']}, role={config.get('role', 'unknown')})")
    
    # Process each flight
    all_results = []
    
    for flight_name, flight_config in FLIGHTS.items():
        result = process_flight(flight_name, flight_config)
        
        if result is not None:
            all_results.append(result)
    
    # Cross-flight comparison
    if all_results:
        comparison_df = create_comparison_table(all_results)
        plot_cross_flight_comparison(comparison_df)
        
        # Final summary
        print("\n" + "=" * 70)
        print("FINAL SUMMARY")
        print("=" * 70)
        
        print("\n📊 Flights Suitable for Main Analysis:")
        suitable_flights = comparison_df[comparison_df["suitable"] == True]
        if len(suitable_flights) > 0:
            for _, row in suitable_flights.iterrows():
                print(f"   ✅ Flight {row['flight']}: "
                      f"{row['duration_s']:.1f} s, "
                      f"{row['n_windows']} windows, "
                      f"{row['quality']}")
        else:
            print("   ❌ No flights fully suitable")
        
        print("\n📊 Flights with Conditional Suitability:")
        conditional = comparison_df[comparison_df["suitable"] == "CONDITIONAL"]
        if len(conditional) > 0:
            for _, row in conditional.iterrows():
                print(f"   ⚠️  Flight {row['flight']}: "
                      f"jitter={row['jitter_std_percent']:.3f}%, "
                      f"gaps={row['gaps_gt_2x']}")
        
        print("\n📊 Flights Not Suitable:")
        not_suitable = comparison_df[comparison_df["suitable"] == False]
        if len(not_suitable) > 0:
            for _, row in not_suitable.iterrows():
                print(f"   ❌ Flight {row['flight']}: "
                      f"jitter={row['jitter_std_percent']:.3f}%, "
                      f"gaps={row['gaps_gt_2x']}")
    
    print("\n" + "=" * 70)
    print("✅ QUALITY AUDIT COMPLETED")
    print("=" * 70)
    
    print(f"\nOutput directory: {BASE_OUTPUT_DIR}")
    print(f"Figures: {BASE_OUTPUT_DIR}/figures")


if __name__ == "__main__":
    main()