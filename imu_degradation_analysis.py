#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MEMS IMU Degradation Analysis
=============================

Pipeline:

PX4 sensor_combined_0
        |
        +--> Timestamp validation
        |
        +--> Accelerometer / Gyroscope extraction
        |
        +--> Sampling-rate validation
        |
        +--> Allan deviation
        |
        +--> Welch PSD
        |
        +--> Vibration metrics
        |
        +--> Windowed degradation features
        |
        +--> Time / Vibration / Time+Vibration models
        |
        +--> CSV + Excel + PNG results

Important scientific notes:
---------------------------
1. No artificial upsampling/interpolation is performed.
2. Sampling frequency is determined from actual timestamps.
3. gyro_rad[] is used directly in rad/s.
4. accelerometer_m_s2[] is used directly in m/s^2.
5. Windowed signal statistics during dynamic flight are NOT automatically
   interpreted as physical IMU bias.
6. High-frequency vibration metrics require a genuinely high-rate signal.
7. Flight 4 is flagged if timestamp irregularity is excessive.
8. Regression coefficients are empirical identification coefficients,
   not validated physical degradation parameters.
"""

# ============================================================
# IMPORTS
# ============================================================

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from scipy import signal
from scipy import stats


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path.home() / "Research" / "validation"

OUTPUT_DIR = BASE_DIR / "imu_degradation_results"
FIGURES_DIR = OUTPUT_DIR / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

FLIGHTS = ["flight2", "flight3", "flight4"]

WINDOW_SECONDS = 30.0

# Minimum number of samples required for a window
MIN_WINDOW_SAMPLES = 100

# PSD configuration
PSD_SEGMENT_SECONDS = 8.0
PSD_OVERLAP = 0.50

# Maximum frequency used in vibration analysis.
# We avoid using exactly Nyquist because numerical estimates
# near Nyquist can be unstable.
MAX_VIBRATION_FRACTION_OF_NYQUIST = 0.90

# Frequency bands
LOW_BAND = (1.0, 10.0)
MID_BAND = (10.0, 50.0)
HIGH_BAND = (50.0, 100.0)

# Gap thresholds
GAP_FACTOR = 2.0
LARGE_GAP_FACTOR = 5.0

# Allan deviation averaging factors
ALLAN_NUM_TAU = 50


# ============================================================
# GENERAL UTILITIES
# ============================================================

def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return np.nan


def rms(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return np.nan

    return float(np.sqrt(np.mean(x ** 2)))


def mad(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return np.nan

    med = np.median(x)
    return float(np.median(np.abs(x - med)))


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)

    if mask.sum() == 0:
        return np.nan

    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))


def mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)

    if mask.sum() == 0:
        return np.nan

    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


def r2_score_manual(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)

    if mask.sum() < 2:
        return np.nan

    yt = y_true[mask]
    yp = y_pred[mask]

    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)

    if ss_tot <= 0:
        return np.nan

    return float(1.0 - ss_res / ss_tot)


def adjusted_r2(r2, n, p):
    if not np.isfinite(r2):
        return np.nan

    if n <= p + 1:
        return np.nan

    return float(
        1.0 - (1.0 - r2) * (n - 1) / (n - p - 1)
    )


def aic_from_residuals(residuals, k):
    residuals = np.asarray(residuals, dtype=float)
    residuals = residuals[np.isfinite(residuals)]

    n = len(residuals)

    if n == 0:
        return np.nan

    rss = np.sum(residuals ** 2)

    if rss <= 0:
        rss = np.finfo(float).eps

    return float(
        n * np.log(rss / n) + 2 * k
    )


def bic_from_residuals(residuals, k):
    residuals = np.asarray(residuals, dtype=float)
    residuals = residuals[np.isfinite(residuals)]

    n = len(residuals)

    if n == 0:
        return np.nan

    rss = np.sum(residuals ** 2)

    if rss <= 0:
        rss = np.finfo(float).eps

    return float(
        n * np.log(rss / n) + k * np.log(n)
    )


def trapezoid_integral(y, x):
    """
    Compatibility wrapper for NumPy versions.
    """

    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))

    return float(np.trapz(y, x))


# ============================================================
# COLUMN IDENTIFICATION
# ============================================================

def identify_timestamp_column(df):
    candidates = [
        "timestamp",
        "timestamp_us",
        "time",
        "time_us",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    # Fallback
    for col in df.columns:
        name = str(col).lower()

        if "timestamp" in name:
            return col

    raise ValueError(
        "Could not identify timestamp column."
    )


def identify_accelerometer_columns(df):
    """
    PX4 sensor_combined format:

        accelerometer_m_s2[0]
        accelerometer_m_s2[1]
        accelerometer_m_s2[2]
    """

    candidates = [

        [
            "accelerometer_m_s2[0]",
            "accelerometer_m_s2[1]",
            "accelerometer_m_s2[2]",
        ],

        [
            "accelerometer_m_s2_0",
            "accelerometer_m_s2_1",
            "accelerometer_m_s2_2",
        ],

        [
            "accel[0]",
            "accel[1]",
            "accel[2]",
        ],

        [
            "acceleration[0]",
            "acceleration[1]",
            "acceleration[2]",
        ],
    ]

    for candidate in candidates:

        if all(col in df.columns for col in candidate):
            return candidate

    # General fallback
    accel_candidates = [
        col for col in df.columns
        if (
            "accelerometer" in str(col).lower()
            or "accel" in str(col).lower()
        )
    ]

    print("Possible accelerometer columns:")
    print(accel_candidates)

    raise ValueError(
        "Could not identify accelerometer columns."
    )


def identify_gyroscope_columns(df):
    """
    PX4 sensor_combined format:

        gyro_rad[0]
        gyro_rad[1]
        gyro_rad[2]

    This is the important correction for the uploaded dataset.
    """

    candidates = [

        # Correct PX4 format found in user's file
        [
            "gyro_rad[0]",
            "gyro_rad[1]",
            "gyro_rad[2]",
        ],

        [
            "gyro[0]",
            "gyro[1]",
            "gyro[2]",
        ],

        [
            "gyro_rad_s[0]",
            "gyro_rad_s[1]",
            "gyro_rad_s[2]",
        ],

        [
            "gyro_rad_s_0",
            "gyro_rad_s_1",
            "gyro_rad_s_2",
        ],

        [
            "gyro_0",
            "gyro_1",
            "gyro_2",
        ],
    ]

    for candidate in candidates:

        if all(col in df.columns for col in candidate):
            return candidate

    # General fallback
    possible_gyro = [
        col for col in df.columns
        if "gyro" in str(col).lower()
    ]

    print("Possible gyro columns found:")
    print(possible_gyro)

    raise ValueError(
        "Could not identify gyroscope columns."
    )


# ============================================================
# DATA LOADING
# ============================================================

def load_flight(flight):
    print_section(f"LOADING {flight.upper()}")

    csv_path = (
        BASE_DIR
        / flight
        / "PX4"
        / "CSV"
        / f"{flight}_sensor_combined_0.csv"
    )

    print(f"File: {csv_path}")

    if not csv_path.exists():
        raise FileNotFoundError(
            f"File not found: {csv_path}"
        )

    df = pd.read_csv(csv_path)

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    timestamp_col = identify_timestamp_column(df)
    accel_cols = identify_accelerometer_columns(df)
    gyro_cols = identify_gyroscope_columns(df)

    print(f"Timestamp: {timestamp_col}")
    print(f"Accelerometer columns: {accel_cols}")
    print(f"Gyroscope columns: {gyro_cols}")

    # Keep required columns
    required = (
        [timestamp_col]
        + accel_cols
        + gyro_cols
    )

    # Optional clipping fields
    optional = []

    for col in [
        "accelerometer_clipping",
        "gyro_clipping",
        "accelerometer_timestamp_relative",
        "gyro_integral_dt",
        "accelerometer_integral_dt",
    ]:

        if col in df.columns:
            optional.append(col)

    df = df[required + optional].copy()

    return df, timestamp_col, accel_cols, gyro_cols


# ============================================================
# TIMESTAMP PROCESSING
# ============================================================

def prepare_data(
    df,
    timestamp_col,
    accel_cols,
    gyro_cols
):

    print_section("TIMESTAMP AND DATA QUALITY")

    work = df.copy()

    # Convert all relevant columns to numeric
    numeric_cols = (
        [timestamp_col]
        + accel_cols
        + gyro_cols
    )

    for col in numeric_cols:
        work[col] = pd.to_numeric(
            work[col],
            errors="coerce"
        )

    original_rows = len(work)

    # Remove NaN
    work = work.dropna(
        subset=numeric_cols
    ).copy()

    removed_nan = (
        original_rows - len(work)
    )

    # Sort
    work = work.sort_values(
        timestamp_col
    ).reset_index(drop=True)

    # Timestamp duplicates
    duplicate_timestamp_count = int(
        work[timestamp_col].duplicated().sum()
    )

    work = work.drop_duplicates(
        subset=[timestamp_col],
        keep="first"
    ).reset_index(drop=True)

    # Timestamp unit detection
    t_raw = work[timestamp_col].to_numpy(
        dtype=float
    )

    if len(t_raw) < 3:
        raise ValueError(
            "Not enough valid timestamped samples."
        )

    median_raw_dt = np.median(
        np.diff(t_raw)
    )

    # PX4 CSV timestamps are usually microseconds.
    # Automatically detect scale.
    if median_raw_dt > 1e3:
        timestamp_scale = 1e-6
        timestamp_unit = "microseconds"
    elif median_raw_dt > 1.0:
        timestamp_scale = 1e-3
        timestamp_unit = "milliseconds"
    else:
        timestamp_scale = 1.0
        timestamp_unit = "seconds"

    t = (
        t_raw * timestamp_scale
    )

    # Normalize time to zero
    t = t - t[0]

    work["time_s"] = t

    dt = np.diff(t)

    # Valid positive intervals
    valid_dt = dt[
        np.isfinite(dt) &
        (dt > 0)
    ]

    if len(valid_dt) == 0:
        raise ValueError(
            "No valid positive timestamp intervals."
        )

    median_dt = float(
        np.median(valid_dt)
    )

    mean_dt = float(
        np.mean(valid_dt)
    )

    std_dt = float(
        np.std(valid_dt)
    )

    mad_dt = float(
        np.median(
            np.abs(
                valid_dt - median_dt
            )
        )
    )

    fs_median = 1.0 / median_dt
    fs_mean = 1.0 / mean_dt

    gap_2x = int(
        np.sum(
            valid_dt >
            GAP_FACTOR * median_dt
        )
    )

    gap_5x = int(
        np.sum(
            valid_dt >
            LARGE_GAP_FACTOR * median_dt
        )
    )

    largest_gap = float(
        np.max(valid_dt)
    )

    jitter_std_percent = (
        100.0 *
        std_dt /
        median_dt
    )

    jitter_mad_percent = (
        100.0 *
        mad_dt /
        median_dt
    )

    duration_s = float(
        t[-1]
    )

    print(f"Timestamp unit detected: {timestamp_unit}")
    print(f"Duration: {duration_s:.3f} s")
    print(f"Median dt: {median_dt * 1000:.4f} ms")
    print(f"Mean dt:   {mean_dt * 1000:.4f} ms")
    print(f"Median fs: {fs_median:.3f} Hz")
    print(f"Mean fs:   {fs_mean:.3f} Hz")
    print(f"Jitter STD: {jitter_std_percent:.3f}%")
    print(f"Jitter MAD: {jitter_mad_percent:.3f}%")
    print(f"Gaps >2x: {gap_2x}")
    print(f"Gaps >5x: {gap_5x}")
    print(
        f"Largest gap: "
        f"{largest_gap * 1000:.3f} ms"
    )

    # Quality classification
    if jitter_std_percent < 5 and gap_2x <= 2:
        quality = "GOOD"

    elif jitter_std_percent < 15 and gap_5x <= 2:
        quality = "ACCEPTABLE"

    else:
        quality = "POOR"

    print(f"Timestamp quality: {quality}")

    quality_summary = {
        "timestamp_unit": timestamp_unit,
        "duration_s": duration_s,
        "samples": len(work),
        "median_dt_s": median_dt,
        "mean_dt_s": mean_dt,
        "fs_median_hz": fs_median,
        "fs_mean_hz": fs_mean,
        "jitter_std_percent": jitter_std_percent,
        "jitter_mad_percent": jitter_mad_percent,
        "gaps_gt_2x": gap_2x,
        "gaps_gt_5x": gap_5x,
        "largest_gap_s": largest_gap,
        "timestamp_quality": quality,
        "removed_nan_rows": removed_nan,
        "duplicate_timestamps": duplicate_timestamp_count,
    }

    # Extract arrays
    accel = work[
        accel_cols
    ].to_numpy(dtype=float)

    gyro = work[
        gyro_cols
    ].to_numpy(dtype=float)

    return (
        work,
        t,
        accel,
        gyro,
        fs_median,
        quality_summary
    )


# ============================================================
# SIGNAL QUALITY
# ============================================================

def signal_quality_report(accel, gyro):

    result = {}

    for name, data, unit in [
        (
            "accel",
            accel,
            "m/s^2"
        ),
        (
            "gyro",
            gyro,
            "rad/s"
        )
    ]:

        for axis in range(3):

            x = data[:, axis]

            result[
                f"{name}_{axis}_mean"
            ] = np.mean(x)

            result[
                f"{name}_{axis}_std"
            ] = np.std(x)

            result[
                f"{name}_{axis}_rms"
            ] = rms(x)

            result[
                f"{name}_{axis}_mad"
            ] = mad(x)

            result[
                f"{name}_{axis}_min"
            ] = np.min(x)

            result[
                f"{name}_{axis}_max"
            ] = np.max(x)

            result[
                f"{name}_{axis}_nan"
            ] = int(
                np.sum(
                    ~np.isfinite(x)
                )
            )

    return result


# ============================================================
# ALLAN DEVIATION
# ============================================================

def allan_deviation(
    x,
    fs,
    num_tau=ALLAN_NUM_TAU
):
    """
    Computes a cluster Allan deviation estimate.

    Input:
        x  : signal
        fs : sampling frequency

    The method uses adjacent non-overlapping cluster means.

    For each averaging factor m:

        y_k = mean of x over m samples

        Allan variance =
            0.5 * mean((y[k+1] - y[k])^2)

    tau = m / fs

    This is appropriate as a practical Allan-deviation
    characterization of the sampled signal.
    """

    x = np.asarray(
        x,
        dtype=float
    )

    x = x[np.isfinite(x)]

    n = len(x)

    if n < 20:
        return (
            np.array([]),
            np.array([])
        )

    max_m = n // 10

    if max_m < 2:
        return (
            np.array([]),
            np.array([])
        )

    # Log-spaced averaging factors
    m_values = np.unique(
        np.logspace(
            0,
            np.log10(max_m),
            num=num_tau
        ).astype(int)
    )

    taus = []
    adevs = []

    for m in m_values:

        if m < 1:
            continue

        n_clusters = n // m

        if n_clusters < 3:
            continue

        trimmed = x[
            :n_clusters * m
        ]

        clusters = trimmed.reshape(
            n_clusters,
            m
        )

        cluster_means = np.mean(
            clusters,
            axis=1
        )

        differences = np.diff(
            cluster_means
        )

        allan_var = (
            0.5 *
            np.mean(
                differences ** 2
            )
        )

        if (
            np.isfinite(allan_var)
            and allan_var > 0
        ):

            tau = m / fs
            adev = np.sqrt(
                allan_var
            )

            taus.append(tau)
            adevs.append(adev)

    return (
        np.asarray(taus),
        np.asarray(adevs)
    )


def allan_characteristic_summary(
    tau,
    adev
):

    if len(tau) == 0:
        return {
            "allan_min": np.nan,
            "allan_min_tau_s": np.nan,
        }

    idx = np.argmin(adev)

    return {
        "allan_min": float(
            adev[idx]
        ),
        "allan_min_tau_s": float(
            tau[idx]
        ),
    }


# ============================================================
# PSD
# ============================================================

def compute_psd(
    x,
    fs,
    segment_seconds=PSD_SEGMENT_SECONDS
):

    x = np.asarray(
        x,
        dtype=float
    )

    x = x[np.isfinite(x)]

    if len(x) < 20:
        return (
            np.array([]),
            np.array([])
        )

    nperseg = int(
        fs * segment_seconds
    )

    nperseg = max(
        256,
        nperseg
    )

    nperseg = min(
        nperseg,
        len(x)
    )

    if nperseg < 16:
        return (
            np.array([]),
            np.array([])
        )

    noverlap = int(
        nperseg * PSD_OVERLAP
    )

    f, pxx = signal.welch(
        x,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="constant",
        scaling="density"
    )

    return f, pxx


# ============================================================
# PSD BAND POWER
# ============================================================

def band_power(
    f,
    pxx,
    low,
    high
):

    if len(f) == 0:
        return np.nan

    mask = (
        (f >= low)
        &
        (f <= high)
    )

    if np.sum(mask) < 2:
        return np.nan

    return trapezoid_integral(
        pxx[mask],
        f[mask]
    )


def band_rms(
    f,
    pxx,
    low,
    high
):

    power = band_power(
        f,
        pxx,
        low,
        high
    )

    if not np.isfinite(power):
        return np.nan

    return float(
        np.sqrt(
            max(power, 0.0)
        )
    )


def psd_summary(
    f,
    pxx,
    fs
):

    if len(f) == 0:
        return {}

    nyquist = fs / 2.0

    maximum_useful_frequency = (
        MAX_VIBRATION_FRACTION_OF_NYQUIST
        * nyquist
    )

    result = {}

    bands = {
        "low_1_10": LOW_BAND,
        "mid_10_50": MID_BAND,
        "high_50_100": HIGH_BAND,
    }

    for name, (
        low,
        high
    ) in bands.items():

        # Do not exceed useful PSD frequency
        high_actual = min(
            high,
            maximum_useful_frequency
        )

        if high_actual <= low:
            power = np.nan
            rms_value = np.nan
        else:
            power = band_power(
                f,
                pxx,
                low,
                high_actual
            )

            rms_value = (
                np.sqrt(
                    max(power, 0.0)
                )
                if np.isfinite(power)
                else np.nan
            )

        result[
            f"{name}_power"
        ] = power

        result[
            f"{name}_rms"
        ] = rms_value

    # Overall vibration band
    vibration_low = 10.0
    vibration_high = maximum_useful_frequency

    if vibration_high > vibration_low:

        vib_power = band_power(
            f,
            pxx,
            vibration_low,
            vibration_high
        )

        vib_rms = (
            np.sqrt(
                max(vib_power, 0.0)
            )
            if np.isfinite(vib_power)
            else np.nan
        )

    else:
        vib_power = np.nan
        vib_rms = np.nan

    result[
        "vibration_10Hz_to_useful_nyquist_power"
    ] = vib_power

    result[
        "vibration_10Hz_to_useful_nyquist_rms"
    ] = vib_rms

    result[
        "nyquist_hz"
    ] = nyquist

    result[
        "maximum_useful_frequency_hz"
    ] = maximum_useful_frequency

    return result


# ============================================================
# FULL-SIGNAL ALLAN + PSD
# ============================================================

def analyze_axis(
    x,
    fs,
    sensor_name,
    axis_name,
    flight
):

    print(
        f"Analyzing "
        f"{flight} "
        f"{sensor_name} "
        f"axis {axis_name}"
    )

    # Allan
    tau, adev = allan_deviation(
        x,
        fs
    )

    allan_summary = (
        allan_characteristic_summary(
            tau,
            adev
        )
    )

    # PSD
    f, pxx = compute_psd(
        x,
        fs
    )

    psd_info = psd_summary(
        f,
        pxx,
        fs
    )

    return {
        "tau": tau,
        "adev": adev,
        "allan_summary": allan_summary,
        "freq": f,
        "psd": pxx,
        "psd_summary": psd_info,
    }


# ============================================================
# WINDOWING
# ============================================================

def generate_windows(
    t,
    window_seconds=WINDOW_SECONDS
):

    if len(t) < 2:
        return []

    duration = t[-1]

    windows = []

    start = 0.0

    while (
        start + window_seconds
        <= duration
    ):

        end = (
            start
            + window_seconds
        )

        mask = (
            (t >= start)
            &
            (t < end)
        )

        if np.sum(mask) > 0:
            windows.append(
                (
                    start,
                    end,
                    mask
                )
            )

        start = end

    # Include last partial window only if
    # it is sufficiently large
    if start < duration:

        mask = t >= start

        if (
            np.sum(mask)
            >= MIN_WINDOW_SAMPLES
        ):

            windows.append(
                (
                    start,
                    duration,
                    mask
                )
            )

    return windows


# ============================================================
# WINDOW FEATURES
# ============================================================

def extract_window_features(
    t,
    accel,
    gyro,
    fs,
    flight
):

    print_section(
        f"WINDOWED FEATURES - {flight.upper()}"
    )

    windows = generate_windows(
        t,
        WINDOW_SECONDS
    )

    print(
        f"Number of windows: "
        f"{len(windows)}"
    )

    rows = []

    for index, (
        start,
        end,
        mask
    ) in enumerate(windows):

        tt = t[mask]

        if len(tt) < MIN_WINDOW_SAMPLES:
            continue

        row = {

            "flight": flight,

            "window_index": index,

            "start_s": start,

            "end_s": end,

            "mid_time_s":
                0.5 * (start + end),

            "samples": len(tt),
        }

        # ----------------------------------------------------
        # Timestamp quality in window
        # ----------------------------------------------------

        if len(tt) >= 3:

            dt = np.diff(tt)

            valid_dt = dt[
                np.isfinite(dt)
                &
                (dt > 0)
            ]

            if len(valid_dt) > 0:

                med_dt = np.median(
                    valid_dt
                )

                row[
                    "window_fs_hz"
                ] = 1.0 / med_dt

                row[
                    "window_dt_jitter_percent"
                ] = (
                    100.0
                    * np.std(valid_dt)
                    / med_dt
                )

                row[
                    "window_gaps_gt_2x"
                ] = int(
                    np.sum(
                        valid_dt
                        > 2 * med_dt
                    )
                )

            else:

                row[
                    "window_fs_hz"
                ] = np.nan

                row[
                    "window_dt_jitter_percent"
                ] = np.nan

                row[
                    "window_gaps_gt_2x"
                ] = np.nan

        # ----------------------------------------------------
        # Accelerometer features
        # ----------------------------------------------------

        for axis in range(3):

            x = accel[mask, axis]

            row[
                f"accel_{axis}_mean"
            ] = np.mean(x)

            row[
                f"accel_{axis}_std"
            ] = np.std(x)

            row[
                f"accel_{axis}_rms"
            ] = rms(x)

            row[
                f"accel_{axis}_mad"
            ] = mad(x)

            row[
                f"accel_{axis}_min"
            ] = np.min(x)

            row[
                f"accel_{axis}_max"
            ] = np.max(x)

            # PSD vibration
            f, pxx = compute_psd(
                x,
                fs
            )

            pinfo = psd_summary(
                f,
                pxx,
                fs
            )

            for key, value in pinfo.items():

                row[
                    f"accel_{axis}_{key}"
                ] = value

        # ----------------------------------------------------
        # Gyroscope features
        # ----------------------------------------------------

        for axis in range(3):

            x = gyro[mask, axis]

            row[
                f"gyro_{axis}_mean"
            ] = np.mean(x)

            row[
                f"gyro_{axis}_std"
            ] = np.std(x)

            row[
                f"gyro_{axis}_rms"
            ] = rms(x)

            row[
                f"gyro_{axis}_mad"
            ] = mad(x)

            row[
                f"gyro_{axis}_min"
            ] = np.min(x)

            row[
                f"gyro_{axis}_max"
            ] = np.max(x)

            # PSD
            f, pxx = compute_psd(
                x,
                fs
            )

            pinfo = psd_summary(
                f,
                pxx,
                fs
            )

            for key, value in pinfo.items():

                row[
                    f"gyro_{axis}_{key}"
                ] = value

        # ----------------------------------------------------
        # Vector norms
        # ----------------------------------------------------

        accel_norm = np.linalg.norm(
            accel[mask],
            axis=1
        )

        gyro_norm = np.linalg.norm(
            gyro[mask],
            axis=1
        )

        row[
            "accel_norm_mean"
        ] = np.mean(accel_norm)

        row[
            "accel_norm_std"
        ] = np.std(accel_norm)

        row[
            "accel_norm_rms"
        ] = rms(accel_norm)

        row[
            "accel_norm_mad"
        ] = mad(accel_norm)

        row[
            "gyro_norm_mean"
        ] = np.mean(gyro_norm)

        row[
            "gyro_norm_std"
        ] = np.std(gyro_norm)

        row[
            "gyro_norm_rms"
        ] = rms(gyro_norm)

        row[
            "gyro_norm_mad"
        ] = mad(gyro_norm)

        # ----------------------------------------------------
        # High-frequency vibration aggregate
        # ----------------------------------------------------

        vibration_accel_values = []

        vibration_gyro_values = []

        for axis in range(3):

            x_acc = accel[
                mask,
                axis
            ]

            x_gyro = gyro[
                mask,
                axis
            ]

            f_a, p_a = compute_psd(
                x_acc,
                fs
            )

            f_g, p_g = compute_psd(
                x_gyro,
                fs
            )

            if len(f_a) > 0:

                high_limit = min(
                    100.0,
                    0.90 * fs / 2.0
                )

                if high_limit > 10.0:

                    power = band_power(
                        f_a,
                        p_a,
                        10.0,
                        high_limit
                    )

                    if np.isfinite(power):
                        vibration_accel_values.append(
                            np.sqrt(
                                max(
                                    power,
                                    0.0
                                )
                            )
                        )

            if len(f_g) > 0:

                high_limit = min(
                    100.0,
                    0.90 * fs / 2.0
                )

                if high_limit > 10.0:

                    power = band_power(
                        f_g,
                        p_g,
                        10.0,
                        high_limit
                    )

                    if np.isfinite(power):
                        vibration_gyro_values.append(
                            np.sqrt(
                                max(
                                    power,
                                    0.0
                                )
                            )
                        )

        if vibration_accel_values:

            row[
                "accel_vibration_rms_mean_axes"
            ] = np.mean(
                vibration_accel_values
            )

        else:

            row[
                "accel_vibration_rms_mean_axes"
            ] = np.nan

        if vibration_gyro_values:

            row[
                "gyro_vibration_rms_mean_axes"
            ] = np.mean(
                vibration_gyro_values
            )

        else:

            row[
                "gyro_vibration_rms_mean_axes"
            ] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# DEGRADATION MODEL
# ============================================================

def fit_regression(
    df,
    y_col,
    feature_columns,
    model_name
):

    required = [
        y_col
    ] + feature_columns

    data = df[
        required
    ].replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    if len(data) < (
        len(feature_columns) + 5
    ):
        return {
            "model": model_name,
            "target": y_col,
            "features": "+".join(
                feature_columns
            ),
            "n": len(data),
            "r2": np.nan,
            "adjusted_r2": np.nan,
            "rmse": np.nan,
            "mae": np.nan,
            "aic": np.nan,
            "bic": np.nan,
            "coefficients": None,
            "intercept": np.nan,
        }

    y = data[
        y_col
    ].to_numpy(
        dtype=float
    )

    X = data[
        feature_columns
    ].to_numpy(
        dtype=float
    )

    # Add intercept
    X_design = np.column_stack(
        [
            np.ones(len(X)),
            X
        ]
    )

    beta, *_ = np.linalg.lstsq(
        X_design,
        y,
        rcond=None
    )

    y_pred = (
        X_design @ beta
    )

    residuals = (
        y - y_pred
    )

    r2 = r2_score_manual(
        y,
        y_pred
    )

    adj = adjusted_r2(
        r2,
        len(y),
        len(feature_columns)
    )

    return {
        "model": model_name,
        "target": y_col,
        "features": "+".join(
            feature_columns
        ),
        "n": len(data),
        "r2": r2,
        "adjusted_r2": adj,
        "rmse": rmse(
            y,
            y_pred
        ),
        "mae": mae(
            y,
            y_pred
        ),
        "aic": aic_from_residuals(
            residuals,
            len(beta)
        ),
        "bic": bic_from_residuals(
            residuals,
            len(beta)
        ),
        "coefficients": beta[1:].tolist(),
        "intercept": float(beta[0]),
    }


def run_degradation_models(
    window_df
):

    print_section(
        "TIME + VIBRATION DEGRADATION MODELS"
    )

    if len(window_df) == 0:
        return pd.DataFrame()

    results = []

    # Targets:
    # signal-level std / RMS and vibration metrics.
    #
    # We deliberately do not call these "physical bias"
    # without stationary/reference validation.

    targets = [

        "accel_0_std",
        "accel_1_std",
        "accel_2_std",

        "gyro_0_std",
        "gyro_1_std",
        "gyro_2_std",

        "accel_vibration_rms_mean_axes",
        "gyro_vibration_rms_mean_axes",
    ]

    # Time variable
    if "mid_time_s" not in window_df.columns:
        return pd.DataFrame()

    # Normalized time for numerical stability
    window_df = window_df.copy()

    max_time = (
        window_df[
            "mid_time_s"
        ].max()
    )

    if (
        np.isfinite(max_time)
        and max_time > 0
    ):

        window_df[
            "time_normalized"
        ] = (
            window_df[
                "mid_time_s"
            ] / max_time
        )

    else:

        window_df[
            "time_normalized"
        ] = 0.0

    vibration_candidates = [

        "accel_vibration_rms_mean_axes",
        "gyro_vibration_rms_mean_axes",
    ]

    # We use the combined vibration metric only
    # if both sensors are available.
    if (
        "accel_vibration_rms_mean_axes"
        in window_df.columns
        and
        "gyro_vibration_rms_mean_axes"
        in window_df.columns
    ):

        window_df[
            "combined_vibration"
        ] = (
            window_df[
                "accel_vibration_rms_mean_axes"
            ]
            +
            window_df[
                "gyro_vibration_rms_mean_axes"
            ]
        ) / 2.0

        vibration_feature = [
            "combined_vibration"
        ]

    elif (
        "accel_vibration_rms_mean_axes"
        in window_df.columns
    ):

        vibration_feature = [
            "accel_vibration_rms_mean_axes"
        ]

    elif (
        "gyro_vibration_rms_mean_axes"
        in window_df.columns
    ):

        vibration_feature = [
            "gyro_vibration_rms_mean_axes"
        ]

    else:

        vibration_feature = []

    for target in targets:

        if target not in window_df.columns:
            continue

        # ---------------------------------------------
        # Time only
        # ---------------------------------------------

        result = fit_regression(
            window_df,
            target,
            ["time_normalized"],
            "time_only"
        )

        results.append(result)

        # ---------------------------------------------
        # Vibration only
        # ---------------------------------------------

        if vibration_feature:

            result = fit_regression(
                window_df,
                target,
                vibration_feature,
                "vibration_only"
            )

            results.append(result)

        # ---------------------------------------------
        # Time + vibration
        # ---------------------------------------------

        if vibration_feature:

            result = fit_regression(
                window_df,
                target,
                [
                    "time_normalized"
                ]
                + vibration_feature,
                "time_plus_vibration"
            )

            results.append(result)

    return pd.DataFrame(results)


# ============================================================
# PLOTS
# ============================================================

def plot_allan(
    flight,
    sensor_name,
    axis_name,
    tau,
    adev
):

    if len(tau) == 0:
        return

    plt.figure(
        figsize=(8, 5)
    )

    plt.loglog(
        tau,
        adev,
        linewidth=1.5
    )

    plt.xlabel(
        "Averaging Time τ (s)"
    )

    if sensor_name == "accelerometer":
        plt.ylabel(
            "Allan Deviation (m/s²)"
        )
    else:
        plt.ylabel(
            "Allan Deviation (rad/s)"
        )

    plt.title(
        f"{flight} - "
        f"{sensor_name} - "
        f"Axis {axis_name}"
    )

    plt.grid(
        True,
        which="both",
        alpha=0.3
    )

    plt.tight_layout()

    path = (
        FIGURES_DIR
        /
        f"{flight}_{sensor_name}_"
        f"axis{axis_name}_allan.png"
    )

    plt.savefig(
        path,
        dpi=180
    )

    plt.close()


def plot_psd(
    flight,
    sensor_name,
    axis_name,
    f,
    pxx
):

    if len(f) == 0:
        return

    mask = (
        (f > 0)
        &
        (pxx > 0)
        &
        np.isfinite(f)
        &
        np.isfinite(pxx)
    )

    if np.sum(mask) < 2:
        return

    plt.figure(
        figsize=(8, 5)
    )

    plt.loglog(
        f[mask],
        pxx[mask],
        linewidth=1.2
    )

    plt.xlabel(
        "Frequency (Hz)"
    )

    if sensor_name == "accelerometer":

        plt.ylabel(
            "PSD ((m/s²)²/Hz)"
        )

    else:

        plt.ylabel(
            "PSD ((rad/s)²/Hz)"
        )

    plt.title(
        f"{flight} - "
        f"{sensor_name} - "
        f"Axis {axis_name} - PSD"
    )

    plt.grid(
        True,
        which="both",
        alpha=0.3
    )

    plt.tight_layout()

    path = (
        FIGURES_DIR
        /
        f"{flight}_{sensor_name}_"
        f"axis{axis_name}_psd.png"
    )

    plt.savefig(
        path,
        dpi=180
    )

    plt.close()


def plot_feature_trends(
    window_df,
    flight
):

    if len(window_df) == 0:
        return

    # --------------------------------------------------------
    # Accelerometer vibration
    # --------------------------------------------------------

    col = (
        "accel_vibration_rms_mean_axes"
    )

    if col in window_df.columns:

        plt.figure(
            figsize=(9, 5)
        )

        plt.plot(
            window_df["mid_time_s"],
            window_df[col],
            marker="o",
            markersize=3,
            linewidth=1
        )

        plt.xlabel(
            "Time (s)"
        )

        plt.ylabel(
            "High-frequency vibration RMS (m/s²)"
        )

        plt.title(
            f"{flight} - "
            "Accelerometer vibration trend"
        )

        plt.grid(
            True,
            alpha=0.3
        )

        plt.tight_layout()

        path = (
            FIGURES_DIR
            /
            f"{flight}_accel_vibration_trend.png"
        )

        plt.savefig(
            path,
            dpi=180
        )

        plt.close()

    # --------------------------------------------------------
    # Gyroscope vibration
    # --------------------------------------------------------

    col = (
        "gyro_vibration_rms_mean_axes"
    )

    if col in window_df.columns:

        plt.figure(
            figsize=(9, 5)
        )

        plt.plot(
            window_df["mid_time_s"],
            window_df[col],
            marker="o",
            markersize=3,
            linewidth=1
        )

        plt.xlabel(
            "Time (s)"
        )

        plt.ylabel(
            "High-frequency vibration RMS (rad/s)"
        )

        plt.title(
            f"{flight} - "
            "Gyroscope vibration trend"
        )

        plt.grid(
            True,
            alpha=0.3
        )

        plt.tight_layout()

        path = (
            FIGURES_DIR
            /
            f"{flight}_gyro_vibration_trend.png"
        )

        plt.savefig(
            path,
            dpi=180
        )

        plt.close()

    # --------------------------------------------------------
    # Standard deviation trend
    # --------------------------------------------------------

    for sensor in [
        "accel",
        "gyro"
    ]:

        for axis in range(3):

            col = (
                f"{sensor}_{axis}_std"
            )

            if col not in window_df.columns:
                continue

            plt.figure(
                figsize=(9, 5)
            )

            plt.plot(
                window_df["mid_time_s"],
                window_df[col],
                marker="o",
                markersize=3,
                linewidth=1
            )

            plt.xlabel(
                "Time (s)"
            )

            if sensor == "accel":

                plt.ylabel(
                    "Standard deviation (m/s²)"
                )

            else:

                plt.ylabel(
                    "Standard deviation (rad/s)"
                )

            plt.title(
                f"{flight} - "
                f"{sensor} axis {axis} "
                "windowed variability"
            )

            plt.grid(
                True,
                alpha=0.3
            )

            plt.tight_layout()

            path = (
                FIGURES_DIR
                /
                f"{flight}_{sensor}_"
                f"axis{axis}_std_trend.png"
            )

            plt.savefig(
                path,
                dpi=180
            )

            plt.close()


# ============================================================
# MAIN FLIGHT ANALYSIS
# ============================================================

def analyze_flight(flight):

    df, timestamp_col, accel_cols, gyro_cols = (
        load_flight(flight)
    )

    (
        clean_df,
        t,
        accel,
        gyro,
        fs,
        quality
    ) = prepare_data(
        df,
        timestamp_col,
        accel_cols,
        gyro_cols
    )

    # --------------------------------------------------------
    # Signal quality
    # --------------------------------------------------------

    signal_quality = (
        signal_quality_report(
            accel,
            gyro
        )
    )

    # --------------------------------------------------------
    # Allan + PSD
    # --------------------------------------------------------

    allan_rows = []
    psd_rows = []

    analysis_store = {}

    for sensor_name, data in [
        (
            "accelerometer",
            accel
        ),
        (
            "gyroscope",
            gyro
        )
    ]:

        for axis in range(3):

            result = analyze_axis(
                data[:, axis],
                fs,
                sensor_name,
                axis,
                flight
            )

            analysis_store[
                (
                    sensor_name,
                    axis
                )
            ] = result

            # ----------------------------------------------
            # Allan result
            # ----------------------------------------------

            tau = result["tau"]
            adev = result["adev"]

            if len(tau) > 0:

                for tt, aa in zip(
                    tau,
                    adev
                ):

                    allan_rows.append({

                        "flight": flight,

                        "sensor": sensor_name,

                        "axis": axis,

                        "tau_s": tt,

                        "allan_deviation": aa,

                    })

            # ----------------------------------------------
            # PSD result
            # ----------------------------------------------

            f = result["freq"]
            pxx = result["psd"]

            if len(f) > 0:

                for ff, pp in zip(
                    f,
                    pxx
                ):

                    psd_rows.append({

                        "flight": flight,

                        "sensor": sensor_name,

                        "axis": axis,

                        "frequency_hz": ff,

                        "psd": pp,

                    })

            # ----------------------------------------------
            # Figures
            # ----------------------------------------------

            plot_allan(
                flight,
                sensor_name,
                axis,
                tau,
                adev
            )

            plot_psd(
                flight,
                sensor_name,
                axis,
                f,
                pxx
            )

    allan_df = pd.DataFrame(
        allan_rows
    )

    psd_df = pd.DataFrame(
        psd_rows
    )

    # --------------------------------------------------------
    # Allan summary
    # --------------------------------------------------------

    allan_summary_rows = []

    for (
        sensor_name,
        axis
    ), result in analysis_store.items():

        row = {

            "flight": flight,

            "sensor": sensor_name,

            "axis": axis,

            **result[
                "allan_summary"
            ],

            **result[
                "psd_summary"
            ],

        }

        allan_summary_rows.append(
            row
        )

    allan_summary_df = pd.DataFrame(
        allan_summary_rows
    )

    # --------------------------------------------------------
    # Window features
    # --------------------------------------------------------

    window_df = extract_window_features(
        t,
        accel,
        gyro,
        fs,
        flight
    )

    # Add timestamp quality
    for key, value in quality.items():

        window_df[
            key
        ] = value

    # --------------------------------------------------------
    # Feature plots
    # --------------------------------------------------------

    plot_feature_trends(
        window_df,
        flight
    )

    # --------------------------------------------------------
    # Regression models
    # --------------------------------------------------------

    degradation_df = (
        run_degradation_models(
            window_df
        )
    )

    # --------------------------------------------------------
    # Full flight summary
    # --------------------------------------------------------

    summary = {

        "flight": flight,

        **quality,

        **signal_quality,

    }

    return {
        "clean_data": clean_df,
        "allan": allan_df,
        "psd": psd_df,
        "allan_summary": allan_summary_df,
        "windows": window_df,
        "degradation": degradation_df,
        "summary": summary,
        "analysis_store": analysis_store,
    }


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    all_results
):

    print_section(
        "SAVING RESULTS"
    )

    summary_rows = []
    allan_list = []
    psd_list = []
    window_list = []
    degradation_list = []
    allan_summary_list = []

    for flight, result in all_results.items():

        summary_rows.append(
            result["summary"]
        )

        if len(
            result["allan"]
        ) > 0:

            allan_list.append(
                result["allan"]
            )

        if len(
            result["psd"]
        ) > 0:

            psd_list.append(
                result["psd"]
            )

        if len(
            result["windows"]
        ) > 0:

            window_list.append(
                result["windows"]
            )

        if len(
            result["degradation"]
        ) > 0:

            degradation_list.append(
                result["degradation"]
            )

        if len(
            result["allan_summary"]
        ) > 0:

            allan_summary_list.append(
                result["allan_summary"]
            )

    # --------------------------------------------------------
    # Concatenate
    # --------------------------------------------------------

    summary_df = pd.DataFrame(
        summary_rows
    )

    allan_df = (
        pd.concat(
            allan_list,
            ignore_index=True
        )
        if allan_list
        else pd.DataFrame()
    )

    psd_df = (
        pd.concat(
            psd_list,
            ignore_index=True
        )
        if psd_list
        else pd.DataFrame()
    )

    window_df = (
        pd.concat(
            window_list,
            ignore_index=True
        )
        if window_list
        else pd.DataFrame()
    )

    degradation_df = (
        pd.concat(
            degradation_list,
            ignore_index=True
        )
        if degradation_list
        else pd.DataFrame()
    )

    allan_summary_df = (
        pd.concat(
            allan_summary_list,
            ignore_index=True
        )
        if allan_summary_list
        else pd.DataFrame()
    )

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    summary_path = (
        OUTPUT_DIR
        / "flight_summary.csv"
    )

    allan_path = (
        OUTPUT_DIR
        / "allan_deviation.csv"
    )

    allan_summary_path = (
        OUTPUT_DIR
        / "allan_psd_summary.csv"
    )

    psd_path = (
        OUTPUT_DIR
        / "psd_results.csv"
    )

    window_path = (
        OUTPUT_DIR
        / "windowed_features.csv"
    )

    degradation_path = (
        OUTPUT_DIR
        / "degradation_models.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False
    )

    allan_df.to_csv(
        allan_path,
        index=False
    )

    allan_summary_df.to_csv(
        allan_summary_path,
        index=False
    )

    psd_df.to_csv(
        psd_path,
        index=False
    )

    window_df.to_csv(
        window_path,
        index=False
    )

    degradation_df.to_csv(
        degradation_path,
        index=False
    )

    # --------------------------------------------------------
    # Excel
    # --------------------------------------------------------

    excel_path = (
        OUTPUT_DIR
        / "imu_degradation_analysis.xlsx"
    )

    with pd.ExcelWriter(
        excel_path,
        engine="openpyxl"
    ) as writer:

        summary_df.to_excel(
            writer,
            sheet_name="Flight_Summary",
            index=False
        )

        allan_summary_df.to_excel(
            writer,
            sheet_name="Allan_PSD_Summary",
            index=False
        )

        allan_df.to_excel(
            writer,
            sheet_name="Allan_Deviation",
            index=False
        )

        psd_df.to_excel(
            writer,
            sheet_name="PSD",
            index=False
        )

        window_df.to_excel(
            writer,
            sheet_name="Window_Features",
            index=False
        )

        degradation_df.to_excel(
            writer,
            sheet_name="Degradation_Models",
            index=False
        )

    print(
        f"Excel saved:\n{excel_path}"
    )

    return {
        "summary": summary_df,
        "allan": allan_df,
        "allan_summary": allan_summary_df,
        "psd": psd_df,
        "windows": window_df,
        "degradation": degradation_df,
    }


# ============================================================
# FINAL REPORT
# ============================================================

def print_final_report(
    saved
):

    print_section(
        "FINAL ANALYSIS SUMMARY"
    )

    print(
        "\nFLIGHT SUMMARY"
    )

    print(
        "-" * 80
    )

    summary = saved[
        "summary"
    ]

    if len(summary) > 0:

        cols = [
            "flight",
            "samples",
            "duration_s",
            "fs_median_hz",
            "jitter_std_percent",
            "gaps_gt_2x",
            "gaps_gt_5x",
            "timestamp_quality",
        ]

        cols = [
            c for c in cols
            if c in summary.columns
        ]

        print(
            summary[
                cols
            ].to_string(
                index=False
            )
        )

    print(
        "\nALLAN / PSD SUMMARY"
    )

    print(
        "-" * 80
    )

    allan_summary = saved[
        "allan_summary"
    ]

    if len(allan_summary) > 0:

        cols = [
            "flight",
            "sensor",
            "axis",
            "allan_min",
            "allan_min_tau_s",
            "low_1_10_power",
            "mid_10_50_power",
            "high_50_100_power",
            "vibration_10Hz_to_useful_nyquist_rms",
        ]

        cols = [
            c for c in cols
            if c in allan_summary.columns
        ]

        print(
            allan_summary[
                cols
            ].to_string(
                index=False
            )
        )

    print(
        "\nTIME + VIBRATION DEGRADATION MODELS"
    )

    print(
        "-" * 80
    )

    degradation = saved[
        "degradation"
    ]

    if len(degradation) > 0:

        cols = [
            "model",
            "target",
            "features",
            "n",
            "r2",
            "adjusted_r2",
            "rmse",
            "mae",
            "aic",
            "bic",
        ]

        cols = [
            c for c in cols
            if c in degradation.columns
        ]

        print(
            degradation[
                cols
            ].to_string(
                index=False
            )
        )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "FILES GENERATED"
    )

    print(
        "=" * 80
    )

    print(
        f"\nResults:\n{OUTPUT_DIR}"
    )

    print(
        f"\nFigures:\n{FIGURES_DIR}"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        """
The degradation coefficients generated here are empirical
identification coefficients.

They should NOT yet be inserted into the EKF as validated
physical degradation parameters.

Before EKF integration, inspect:

    1. Allan deviation curves
    2. PSD curves
    3. Windowed feature trends
    4. Time-only regression
    5. Vibration-only regression
    6. Time + vibration regression
    7. Flight-to-flight consistency
    8. Flight 4 timestamp quality
    9. Sensor units
   10. Stationary/reference segments
   11. Temperature availability
   12. Clipping indicators

Also remember:

    Windowed mean/std/RMS during dynamic flight are
    signal-level statistics and are not automatically
    equivalent to physical sensor bias or bias drift.
"""
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print_section(
        "MEMS IMU DEGRADATION ANALYSIS"
    )

    print(
        f"\nBase directory:\n{BASE_DIR}"
    )

    print(
        f"\nOutput directory:\n{OUTPUT_DIR}"
    )

    all_results = {}

    for flight in FLIGHTS:

        try:

            result = analyze_flight(
                flight
            )

            all_results[
                flight
            ] = result

            print(
                f"\n{flight.upper()} "
                f"analysis completed successfully."
            )

        except Exception as exc:

            print(
                "\n"
                + "!" * 80
            )

            print(
                f"ERROR processing {flight}:"
            )

            print(
                str(exc)
            )

            print(
                "!" * 80
            )

    if not all_results:

        raise RuntimeError(
            "No flight was successfully analyzed."
        )

    saved = save_results(
        all_results
    )

    print_final_report(
        saved
    )

    print(
        "\nAnalysis completed."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    warnings.filterwarnings(
        "ignore",
        category=RuntimeWarning
    )

    main()