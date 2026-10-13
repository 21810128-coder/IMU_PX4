#!/usr/bin/env python3

"""
IMU PSD Analysis for PX4 SensorCombined Logs
---------------------------------------------

Purpose:
    Stage 2 of the MEMS IMU degradation analysis.

    Analyze frequency content of accelerometer and gyroscope signals
    for Flight 2, Flight 3, and Flight 5 using Welch PSD.

Important:
    - No interpolation
    - No resampling
    - 30-second complete windows
    - Sampling frequency estimated from each window
    - Welch PSD used only on windows with acceptable timestamp quality
    - 10 Hz is NOT assumed to be the final vibration cutoff
    - Candidate frequency bands are exploratory
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import welch


# ============================================================
# CONFIGURATION
# ============================================================

FLIGHTS = {
    "F2": Path(
        "/home/fatima/Research/validation/flight2/PX4/CSV/"
        "flight2_sensor_combined_0.csv"
    ),
    "F3": Path(
        "/home/fatima/Research/validation/flight3/PX4/CSV/"
        "flight3_sensor_combined_0.csv"
    ),
    "F5": Path(
        "/home/fatima/Research/validation/flight5/PX4/CSV/"
        "flight5_sensor_combined_0.csv"
    ),
}

OUTPUT_DIR = Path(
    "/home/fatima/Research/validation/"
    "imu_degradation_results/window_feature_analysis/psd_analysis"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_SECONDS = 30.0
N_PER_SEG = 4096
NO_OVERLAP = N_PER_SEG // 2

MIN_FS = 180.0
MAX_FS = 220.0

MAX_JITTER_STD_PERCENT = 5.0
MAX_GAP_MULTIPLIER = 2.0

BANDS = {
    "0_1Hz": (0.0, 1.0),
    "1_5Hz": (1.0, 5.0),
    "5_10Hz": (5.0, 10.0),
    "10_20Hz": (10.0, 20.0),
    "20_50Hz": (20.0, 50.0),
    "50_100Hz": (50.0, 100.0),
    "100_Nyquist": (100.0, np.inf),
}

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

ACCEL_NAMES = ["accel_x", "accel_y", "accel_z"]
GYRO_NAMES = ["gyro_x", "gyro_y", "gyro_z"]


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def estimate_sampling_quality(timestamp_us):
    """Estimate sampling frequency and timestamp quality."""
    t = np.asarray(timestamp_us, dtype=np.float64) / 1e6
    dt = np.diff(t)
    valid_dt = dt[np.isfinite(dt) & (dt > 0)]

    if len(valid_dt) == 0:
        return {
            "fs_median": np.nan,
            "fs_mean": np.nan,
            "dt_median": np.nan,
            "dt_mean": np.nan,
            "jitter_std_percent": np.nan,
            "jitter_mad_percent": np.nan,
            "gaps_gt_2x": np.nan,
            "gaps_gt_5x": np.nan,
            "largest_gap_ms": np.nan,
            "negative_dt": int(np.sum(dt < 0)),
            "zero_dt": int(np.sum(dt == 0)),
        }

    median_dt = np.median(valid_dt)
    mean_dt = np.mean(valid_dt)

    fs_median = 1.0 / median_dt
    fs_mean = 1.0 / mean_dt

    jitter_percent = (valid_dt - median_dt) / median_dt * 100.0
    jitter_std_percent = np.std(jitter_percent)

    mad = np.median(np.abs(jitter_percent - np.median(jitter_percent)))

    gaps_gt_2x = int(np.sum(valid_dt > 2.0 * median_dt))
    gaps_gt_5x = int(np.sum(valid_dt > 5.0 * median_dt))
    largest_gap_ms = np.max(valid_dt) * 1000.0

    return {
        "fs_median": fs_median,
        "fs_mean": fs_mean,
        "dt_median": median_dt,
        "dt_mean": mean_dt,
        "jitter_std_percent": jitter_std_percent,
        "jitter_mad_percent": mad,
        "gaps_gt_2x": gaps_gt_2x,
        "gaps_gt_5x": gaps_gt_5x,
        "largest_gap_ms": largest_gap_ms,
        "negative_dt": int(np.sum(dt < 0)),
        "zero_dt": int(np.sum(dt == 0)),
    }


def is_window_good(timestamp_us):
    """Decide whether a window is acceptable for PSD."""
    quality = estimate_sampling_quality(timestamp_us)

    if not np.isfinite(quality["fs_median"]):
        return False, quality

    if not (MIN_FS <= quality["fs_median"] <= MAX_FS):
        return False, quality

    if quality["jitter_std_percent"] > MAX_JITTER_STD_PERCENT:
        return False, quality

    if quality["gaps_gt_2x"] > 0:
        return False, quality

    if quality["negative_dt"] > 0:
        return False, quality

    if quality["zero_dt"] > 0:
        return False, quality

    return True, quality


def get_complete_windows(df):
    """Split dataframe into complete 30-second timestamp-based windows."""
    t = df["timestamp"].to_numpy(dtype=np.float64) / 1e6
    start_time = t[0]
    end_time = t[-1]

    n_complete = int(np.floor((end_time - start_time) / WINDOW_SECONDS))
    windows = []

    for i in range(n_complete):
        window_start = start_time + i * WINDOW_SECONDS
        window_end = window_start + WINDOW_SECONDS

        mask = (t >= window_start) & (t < window_end)
        indices = np.flatnonzero(mask)

        if len(indices) == 0:
            continue

        actual_duration = t[indices[-1]] - t[indices[0]]

        if actual_duration < 0.98 * WINDOW_SECONDS:
            continue

        windows.append({
            "window_id": i,
            "start_time_s": window_start,
            "end_time_s": window_end,
            "indices": indices,
            "n_samples": len(indices),
        })

    return windows


def compute_psd(signal, fs):
    """Compute Welch PSD."""
    x = np.asarray(signal, dtype=np.float64)
    x = x[np.isfinite(x)]

    if len(x) < 32:
        return None, None

    nperseg = min(N_PER_SEG, len(x))
    noverlap = min(NO_OVERLAP, nperseg // 2)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f, psd = welch(
            x,
            fs=fs,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            detrend="constant",
            scaling="density",
            average="median",
        )

    return f, psd


def integrate_band_power(f, psd, low, high):
    """Integrate PSD over a frequency band."""
    if f is None or psd is None:
        return np.nan

    upper = min(high, np.max(f))
    mask = (f >= low) & (f < upper)

    if np.sum(mask) < 2:
        return np.nan

    return np.trapezoid(psd[mask], f[mask])


def find_dominant_peak(f, psd, min_frequency=0.5):
    """Find dominant PSD peak above min_frequency."""
    if f is None or psd is None:
        return np.nan, np.nan

    mask = np.isfinite(f) & np.isfinite(psd) & (f >= min_frequency)

    if np.sum(mask) == 0:
        return np.nan, np.nan

    f_valid = f[mask]
    psd_valid = psd[mask]
    idx = np.argmax(psd_valid)

    return float(f_valid[idx]), float(psd_valid[idx])


def rms_from_psd(f, psd, low=0.0, high=np.inf):
    """Calculate RMS equivalent from integrated PSD power."""
    power = integrate_band_power(f, psd, low, high)

    if not np.isfinite(power):
        return np.nan
    if power < 0:
        return np.nan

    return np.sqrt(power)


# ============================================================
# PLOTTING - FIXED VERSION
# ============================================================

def plot_median_psd(median_psd_df, flight, sensor_type, axes_names, output_path):
    """
    Plot median PSD for the three axes.
    
    FIXED: Uses 'frequency_hz' column (common for all axes).
    """
    plt.figure(figsize=(10, 6))

    # Check if frequency column exists
    if "frequency_hz" not in median_psd_df.columns:
        print(f"⚠️ 'frequency_hz' column not found in median_psd_df")
        print(f"   Available columns: {median_psd_df.columns.tolist()}")
        plt.close()
        return

    # Get frequency vector (common for all axes)
    f = median_psd_df["frequency_hz"].to_numpy()

    for axis_name in axes_names:
        psd_col = f"{axis_name}_psd"
        
        if psd_col not in median_psd_df.columns:
            print(f"⚠️ Column '{psd_col}' not found, skipping {axis_name}")
            continue

        psd = median_psd_df[psd_col].to_numpy()

        mask = (
            np.isfinite(f)
            & np.isfinite(psd)
            & (f > 0)
            & (psd > 0)
        )

        if np.sum(mask) == 0:
            print(f"⚠️ No valid PSD data for {axis_name}")
            continue

        plt.semilogy(
            f[mask],
            psd[mask],
            label=axis_name,
            linewidth=1.5,
        )

    # Candidate boundaries only
    for boundary in [1, 5, 10, 20, 50, 100]:
        plt.axvline(
            boundary,
            linestyle="--",
            linewidth=0.8,
            alpha=0.5,
        )

    plt.xlabel("Frequency [Hz]", fontsize=12)

    if sensor_type == "accel":
        plt.ylabel("PSD [(m/s²)²/Hz]", fontsize=12)
        title = f"{flight} - Accelerometer Median PSD"
    else:
        plt.ylabel("PSD [(rad/s)²/Hz]", fontsize=12)
        title = f"{flight} - Gyroscope Median PSD"

    plt.title(title, fontsize=14, fontweight="bold")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend(fontsize=11)
    plt.xlim(0.1, 100.0)
    plt.tight_layout()

    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    
    print(f"✅ Figure saved: {output_path}")


# ============================================================
# MAIN FLIGHT PROCESSING
# ============================================================

def process_flight(flight_name, csv_path):
    """Process a single flight."""

    print("\n" + "=" * 70)
    print(f"Processing {flight_name}")
    print("=" * 70)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found:\n{csv_path}")

    print(f"Loading:\n{csv_path}")

    df = pd.read_csv(csv_path)

    required_columns = ["timestamp"] + ACCEL_COLS + GYRO_COLS

    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {flight_name}: {missing}")

    df = df[required_columns].copy()
    df = df[np.isfinite(df["timestamp"])].copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    print(f"Rows: {len(df):,}")

    windows = get_complete_windows(df)
    print(f"Complete 30-s windows: {len(windows)}")

    bandpower_records = []
    peak_records = []

    psd_storage = {
        "accel_x": [],
        "accel_y": [],
        "accel_z": [],
        "gyro_x": [],
        "gyro_y": [],
        "gyro_z": [],
    }

    frequency_storage = None
    valid_window_count = 0

    for window in windows:

        indices = window["indices"]
        win = df.iloc[indices]
        timestamp = win["timestamp"].to_numpy()

        good, quality = is_window_good(timestamp)

        if not good:
            continue

        fs = quality["fs_median"]
        valid_window_count += 1

        common_metadata = {
            "flight": flight_name,
            "window_id": window["window_id"],
            "start_time_s": window["start_time_s"],
            "end_time_s": window["end_time_s"],
            "n_samples": window["n_samples"],
            "fs_hz": fs,
            "jitter_std_percent": quality["jitter_std_percent"],
            "gaps_gt_2x": quality["gaps_gt_2x"],
        }

        # Process accelerometer
        for column, axis_name in zip(ACCEL_COLS, ACCEL_NAMES):

            x = win[column].to_numpy(dtype=np.float64)
            f, psd = compute_psd(x, fs)

            if f is None:
                continue

            if frequency_storage is None:
                frequency_storage = f.copy()

            if len(f) == len(frequency_storage):
                psd_storage[axis_name].append(psd)

            peak_freq, peak_psd = find_dominant_peak(f, psd, min_frequency=0.5)

            peak_record = dict(common_metadata)
            peak_record.update({
                "sensor": "accel",
                "axis": axis_name,
                "dominant_frequency_hz": peak_freq,
                "dominant_psd": peak_psd,
            })
            peak_records.append(peak_record)

            band_record = dict(common_metadata)
            band_record.update({
                "sensor": "accel",
                "axis": axis_name,
            })

            for band_name, (low, high) in BANDS.items():
                band_record[f"{band_name}_power"] = integrate_band_power(f, psd, low, high)
                band_record[f"{band_name}_rms"] = rms_from_psd(f, psd, low, high)

            bandpower_records.append(band_record)

        # Process gyroscope
        for column, axis_name in zip(GYRO_COLS, GYRO_NAMES):

            x = win[column].to_numpy(dtype=np.float64)
            f, psd = compute_psd(x, fs)

            if f is None:
                continue

            if frequency_storage is None:
                frequency_storage = f.copy()

            if len(f) == len(frequency_storage):
                psd_storage[axis_name].append(psd)

            peak_freq, peak_psd = find_dominant_peak(f, psd, min_frequency=0.5)

            peak_record = dict(common_metadata)
            peak_record.update({
                "sensor": "gyro",
                "axis": axis_name,
                "dominant_frequency_hz": peak_freq,
                "dominant_psd": peak_psd,
            })
            peak_records.append(peak_record)

            band_record = dict(common_metadata)
            band_record.update({
                "sensor": "gyro",
                "axis": axis_name,
            })

            for band_name, (low, high) in BANDS.items():
                band_record[f"{band_name}_power"] = integrate_band_power(f, psd, low, high)
                band_record[f"{band_name}_rms"] = rms_from_psd(f, psd, low, high)

            bandpower_records.append(band_record)

    print(f"Valid PSD windows: {valid_window_count}")

    # ========================================================
    # Median PSD - FIXED COLUMN NAMES
    # ========================================================

    if frequency_storage is not None:
        median_data = {
            "frequency_hz": frequency_storage
        }

        for axis_name in (ACCEL_NAMES + GYRO_NAMES):
            spectra = psd_storage[axis_name]

            if len(spectra) == 0:
                median_data[f"{axis_name}_psd"] = np.full(
                    len(frequency_storage), np.nan
                )
            else:
                values = np.asarray(spectra)
                median_data[f"{axis_name}_psd"] = np.median(values, axis=0)

        median_psd_df = pd.DataFrame(median_data)
    else:
        median_psd_df = pd.DataFrame()

    # ========================================================
    # Save median PSD
    # ========================================================

    median_path = OUTPUT_DIR / f"{flight_name.lower()}_median_psd.csv"
    median_psd_df.to_csv(median_path, index=False)
    print(f"✅ Median PSD saved: {median_path}")

    # ========================================================
    # Save figures
    # ========================================================

    if not median_psd_df.empty:
        plot_median_psd(
            median_psd_df,
            flight_name,
            "accel",
            ACCEL_NAMES,
            OUTPUT_DIR / f"{flight_name.lower()}_accel_psd.png",
        )

        plot_median_psd(
            median_psd_df,
            flight_name,
            "gyro",
            GYRO_NAMES,
            OUTPUT_DIR / f"{flight_name.lower()}_gyro_psd.png",
        )

    # ========================================================
    # Summary
    # ========================================================

    summary = {
        "flight": flight_name,
        "total_rows": len(df),
        "total_complete_windows": len(windows),
        "valid_psd_windows": valid_window_count,
    }

    if len(df) > 1:
        quality = estimate_sampling_quality(df["timestamp"].to_numpy())
        summary.update({
            "global_fs_median_hz": quality["fs_median"],
            "global_fs_mean_hz": quality["fs_mean"],
            "global_jitter_std_percent": quality["jitter_std_percent"],
            "global_jitter_mad_percent": quality["jitter_mad_percent"],
            "global_gaps_gt_2x": quality["gaps_gt_2x"],
            "global_gaps_gt_5x": quality["gaps_gt_5x"],
            "global_largest_gap_ms": quality["largest_gap_ms"],
        })

    return (
        summary,
        pd.DataFrame(bandpower_records),
        pd.DataFrame(peak_records),
        median_psd_df,
    )


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("IMU PSD ANALYSIS - F2 / F3 / F5")
    print("=" * 70)

    all_summaries = []
    all_bandpower = []
    all_peaks = []

    for flight_name, csv_path in FLIGHTS.items():

        (summary, bandpower, peaks, median_psd) = process_flight(
            flight_name, csv_path
        )

        all_summaries.append(summary)

        if not bandpower.empty:
            all_bandpower.append(bandpower)

        if not peaks.empty:
            all_peaks.append(peaks)

    # Combine outputs
    summary_df = pd.DataFrame(all_summaries)

    bandpower_df = (
        pd.concat(all_bandpower, ignore_index=True)
        if all_bandpower else pd.DataFrame()
    )

    peaks_df = (
        pd.concat(all_peaks, ignore_index=True)
        if all_peaks else pd.DataFrame()
    )

    # Save
    summary_path = OUTPUT_DIR / "flight2_3_5_psd_summary.csv"
    bandpower_path = OUTPUT_DIR / "flight2_3_5_psd_bandpower.csv"
    peaks_path = OUTPUT_DIR / "flight2_3_5_psd_peaks.csv"

    summary_df.to_csv(summary_path, index=False)
    bandpower_df.to_csv(bandpower_path, index=False)
    peaks_df.to_csv(peaks_path, index=False)

    # Print summary
    print("\n")
    print("=" * 70)
    print("PSD ANALYSIS COMPLETE")
    print("=" * 70)

    print(f"\nOutput directory:\n{OUTPUT_DIR}")

    print("\nSummary:")
    print(summary_df.to_string(index=False))

    if not bandpower_df.empty:
        print(f"\nBand-power rows: {len(bandpower_df)}")
        print("\nSensors:")
        print(
            bandpower_df[["flight", "sensor", "axis"]]
            .drop_duplicates()
            .to_string(index=False)
        )

    if not peaks_df.empty:
        print("\nDominant-frequency summary:")
        peak_summary = (
            peaks_df
            .groupby(["flight", "sensor", "axis"])["dominant_frequency_hz"]
            .agg(["median", "mean", "min", "max"])
            .reset_index()
        )
        print(peak_summary.to_string(index=False))

    print(f"\nSaved files:")
    print(f"  {summary_path}")
    print(f"  {bandpower_path}")
    print(f"  {peaks_path}")

    print("\nFigures:")
    for f in ["flight2", "flight3", "flight5"]:
        print(f"  {OUTPUT_DIR}/{f}_accel_psd.png")
        print(f"  {OUTPUT_DIR}/{f}_gyro_psd.png")

    print("\nNext step:")
    print("  Inspect PSD frequency content before selecting "
          "the final vibration frequency band.")


if __name__ == "__main__":
    main()