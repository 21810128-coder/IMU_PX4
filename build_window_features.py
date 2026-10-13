#!/usr/bin/env python3

"""
Build 30-second IMU feature dataset for Flights 2, 3 and 5.

Purpose
-------
Stage 1 of the degradation-analysis pipeline.

This script:
    1. Loads PX4 sensor_combined_0.csv
    2. Performs timestamp quality control
    3. Checks clipping/calibration fields when available
    4. Divides each flight into 30-second windows
    5. Computes IMU variability features
    6. Separates low-frequency and high-frequency components
    7. Computes independent vibration RMS features
    8. Saves one feature table per flight
    9. Saves one combined F2/F3/F5 dataset

Important
---------
NO interpolation
NO resampling
NO target-derived vibration predictor
NO degradation conclusion is made here

The high-frequency vibration feature is derived from the IMU signal
within each window using a zero-phase Butterworth high-pass filter.
The cutoff is a configurable analysis parameter and should be
validated later using PSD results.

Expected PX4 columns:
    timestamp
    accelerometer_m_s2[0:2]
    gyro_rad[0:2]

Optional PX4 columns:
    accelerometer_clipping
    gyro_clipping
    accel_calibration_count
    gyro_calibration_count
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from scipy.signal import butter, sosfiltfilt


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path("/home/fatima/Research/validation")

FLIGHTS = [2, 3, 5]

OUTPUT_DIR = (
    BASE_DIR
    / "imu_degradation_results"
    / "window_feature_analysis"
)

WINDOW_SEC = 30.0

EXPECTED_FS = 200.0

# ------------------------------------------------------------
# Vibration filter
# ------------------------------------------------------------
#
# Initial value only.
# We will validate this later against Welch PSD.
#
# Frequencies below this cutoff are treated as
# low-frequency motion/gravity-related content.
#
VIBRATION_HIGHPASS_HZ = 10.0

FILTER_ORDER = 4

# ------------------------------------------------------------
# Timestamp quality
# ------------------------------------------------------------

GOOD_JITTER_STD_PERCENT = 2.0
WARNING_JITTER_STD_PERCENT = 5.0


# ============================================================
# COLUMNS
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

ACCEL_NAMES = ["accel_x", "accel_y", "accel_z"]
GYRO_NAMES = ["gyro_x", "gyro_y", "gyro_z"]

REQUIRED_COLS = (
    ["timestamp"]
    + ACCEL_COLS
    + GYRO_COLS
)

OPTIONAL_COLS = [
    "accelerometer_clipping",
    "gyro_clipping",
    "accel_calibration_count",
    "gyro_calibration_count",
]


# ============================================================
# BASIC FUNCTIONS
# ============================================================

def rms(x):
    """Root mean square."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return np.nan

    return float(np.sqrt(np.mean(x ** 2)))


def mad(x):
    """Median absolute deviation."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return np.nan

    med = np.median(x)

    return float(
        np.median(np.abs(x - med))
    )


def safe_std(x):
    """Sample standard deviation."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) < 2:
        return np.nan

    return float(np.std(x, ddof=1))


def safe_mean(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return np.nan

    return float(np.mean(x))


# ============================================================
# TIMESTAMP QUALITY
# ============================================================

def timestamp_quality(timestamp_us):
    """
    Calculate timestamp quality without resampling.
    """

    timestamp_us = np.asarray(
        timestamp_us,
        dtype=np.float64
    )

    finite = np.isfinite(timestamp_us)

    timestamp_us = timestamp_us[finite]

    if len(timestamp_us) < 2:
        raise ValueError(
            "Not enough valid timestamps."
        )

    dt_s = np.diff(timestamp_us) * 1e-6

    negative_count = int(
        np.sum(dt_s < 0)
    )

    zero_count = int(
        np.sum(dt_s == 0)
    )

    positive_dt = dt_s[dt_s > 0]

    if len(positive_dt) == 0:
        raise ValueError(
            "No positive timestamp intervals."
        )

    median_dt = np.median(positive_dt)

    mean_dt = np.mean(positive_dt)

    fs = 1.0 / positive_dt

    jitter = (
        (positive_dt - median_dt)
        / median_dt
        * 100.0
    )

    jitter_std = float(
        np.std(jitter)
    )

    jitter_mad = float(
        np.median(
            np.abs(
                jitter - np.median(jitter)
            )
        )
    )

    gaps_2x = int(
        np.sum(
            positive_dt > 2.0 * median_dt
        )
    )

    gaps_5x = int(
        np.sum(
            positive_dt > 5.0 * median_dt
        )
    )

    duration = (
        timestamp_us[-1]
        - timestamp_us[0]
    ) * 1e-6

    if (
        jitter_std <= GOOD_JITTER_STD_PERCENT
        and gaps_2x == 0
    ):
        quality = "GOOD"

    elif (
        jitter_std <= WARNING_JITTER_STD_PERCENT
        and gaps_5x <= 1
    ):
        quality = "WARNING"

    else:
        quality = "POOR"

    return {
        "duration_s": float(duration),
        "median_dt_s": float(median_dt),
        "mean_dt_s": float(mean_dt),
        "median_fs_hz": float(np.median(fs)),
        "mean_fs_hz": float(np.mean(fs)),
        "jitter_std_percent": jitter_std,
        "jitter_mad_percent": jitter_mad,
        "gaps_gt_2x": gaps_2x,
        "gaps_gt_5x": gaps_5x,
        "largest_gap_s": float(np.max(positive_dt)),
        "negative_dt_count": negative_count,
        "zero_dt_count": zero_count,
        "quality": quality,
    }


# ============================================================
# VIBRATION FILTER
# ============================================================

def design_highpass(fs, cutoff_hz, order=4):
    """
    Design zero-phase-compatible SOS high-pass filter.
    """

    if cutoff_hz <= 0:
        raise ValueError(
            "High-pass cutoff must be > 0 Hz."
        )

    if cutoff_hz >= fs / 2:
        raise ValueError(
            "High-pass cutoff must be below Nyquist."
        )

    return butter(
        order,
        cutoff_hz,
        btype="highpass",
        fs=fs,
        output="sos",
    )


def highpass_signal(x, sos):
    """
    Zero-phase high-pass filtering.

    No interpolation or resampling.
    """

    x = np.asarray(x, dtype=float)

    finite = np.isfinite(x)

    if not np.all(finite):

        # For the present dataset we expect no NaNs.
        # Do not interpolate.
        raise ValueError(
            "NaN/Inf detected in signal. "
            "Interpolation is intentionally disabled."
        )

    # scipy's filtfilt requires enough samples.
    pad_length = 3 * (
        2 * len(sos) + 1
    )

    if len(x) <= pad_length:
        raise ValueError(
            "Window too short for zero-phase filter."
        )

    return sosfiltfilt(
        sos,
        x
    )


# ============================================================
# CLIPPING / CALIBRATION
# ============================================================

def clipping_counts(values):
    """
    PX4 clipping bitfield:

        X = 1
        Y = 2
        Z = 4
    """

    values = pd.to_numeric(
        values,
        errors="coerce"
    ).fillna(0).astype(np.uint8)

    return {
        "clip_x": int(
            np.sum((values & 1) != 0)
        ),
        "clip_y": int(
            np.sum((values & 2) != 0)
        ),
        "clip_z": int(
            np.sum((values & 4) != 0)
        ),
        "clip_any": int(
            np.sum(values != 0)
        ),
    }


def calibration_info(
    accel_values,
    gyro_values
):
    """
    Detect calibration counter changes.
    """

    result = {
        "accel_calibration_start": np.nan,
        "accel_calibration_end": np.nan,
        "accel_calibration_changes": np.nan,
        "gyro_calibration_start": np.nan,
        "gyro_calibration_end": np.nan,
        "gyro_calibration_changes": np.nan,
    }

    if accel_values is not None:

        values = pd.to_numeric(
            accel_values,
            errors="coerce"
        ).dropna().to_numpy()

        if len(values) > 0:

            result[
                "accel_calibration_start"
            ] = int(values[0])

            result[
                "accel_calibration_end"
            ] = int(values[-1])

            result[
                "accel_calibration_changes"
            ] = int(
                np.sum(
                    np.diff(values) != 0
                )
            )

    if gyro_values is not None:

        values = pd.to_numeric(
            gyro_values,
            errors="coerce"
        ).dropna().to_numpy()

        if len(values) > 0:

            result[
                "gyro_calibration_start"
            ] = int(values[0])

            result[
                "gyro_calibration_end"
            ] = int(values[-1])

            result[
                "gyro_calibration_changes"
            ] = int(
                np.sum(
                    np.diff(values) != 0
                )
            )

    return result


# ============================================================
# WINDOW FEATURE EXTRACTION
# ============================================================

def extract_window_features(
    df,
    flight_id,
    window_id,
    start_idx,
    end_idx,
    sos_accel,
    sos_gyro,
):
    """
    Extract features from one 30-second window.
    """

    window = df.iloc[
        start_idx:end_idx
    ].copy()

    timestamp = (
        window["timestamp"]
        .to_numpy(dtype=float)
    )

    if len(timestamp) < 100:
        return None

    time_start = (
        timestamp[0]
        - df["timestamp"].iloc[0]
    ) * 1e-6

    time_end = (
        timestamp[-1]
        - df["timestamp"].iloc[0]
    ) * 1e-6

    # --------------------------------------------------------
    # Timestamp quality
    # --------------------------------------------------------

    tq = timestamp_quality(
        timestamp
    )

    # --------------------------------------------------------
    # Base metadata
    # --------------------------------------------------------

    result = {
        "flight": flight_id,
        "window": window_id,

        "start_s": float(time_start),
        "end_s": float(time_end),
        "duration_s": float(
            time_end - time_start
        ),

        "n_samples": len(window),

        "median_dt_s":
            tq["median_dt_s"],

        "mean_dt_s":
            tq["mean_dt_s"],

        "median_fs_hz":
            tq["median_fs_hz"],

        "mean_fs_hz":
            tq["mean_fs_hz"],

        "jitter_std_percent":
            tq["jitter_std_percent"],

        "jitter_mad_percent":
            tq["jitter_mad_percent"],

        "gaps_gt_2x":
            tq["gaps_gt_2x"],

        "gaps_gt_5x":
            tq["gaps_gt_5x"],

        "largest_gap_s":
            tq["largest_gap_s"],

        "timestamp_quality":
            tq["quality"],
    }

    # --------------------------------------------------------
    # Accel features
    # --------------------------------------------------------

    for col, name, sos in zip(
        ACCEL_COLS,
        ACCEL_NAMES,
        [sos_accel] * 3,
    ):

        x = window[col].to_numpy(
            dtype=float
        )

        x_hp = highpass_signal(
            x,
            sos
        )

        result[
            f"{name}_mean"
        ] = safe_mean(x)

        result[
            f"{name}_std"
        ] = safe_std(x)

        result[
            f"{name}_mad"
        ] = mad(x)

        result[
            f"{name}_rms"
        ] = rms(x)

        result[
            f"{name}_highpass_rms"
        ] = rms(x_hp)

        result[
            f"{name}_highpass_std"
        ] = safe_std(x_hp)

        result[
            f"{name}_low_component_rms"
        ] = rms(
            x - x_hp
        )

    # --------------------------------------------------------
    # Gyro features
    # --------------------------------------------------------

    for col, name, sos in zip(
        GYRO_COLS,
        GYRO_NAMES,
        [sos_gyro] * 3,
    ):

        x = window[col].to_numpy(
            dtype=float
        )

        x_hp = highpass_signal(
            x,
            sos
        )

        result[
            f"{name}_mean"
        ] = safe_mean(x)

        result[
            f"{name}_std"
        ] = safe_std(x)

        result[
            f"{name}_mad"
        ] = mad(x)

        result[
            f"{name}_rms"
        ] = rms(x)

        result[
            f"{name}_highpass_rms"
        ] = rms(x_hp)

        result[
            f"{name}_highpass_std"
        ] = safe_std(x_hp)

        result[
            f"{name}_low_component_rms"
        ] = rms(
            x - x_hp
        )

    # --------------------------------------------------------
    # Aggregate vibration metrics
    # --------------------------------------------------------

    accel_vib = np.array([
        result["accel_x_highpass_rms"],
        result["accel_y_highpass_rms"],
        result["accel_z_highpass_rms"],
    ])

    gyro_vib = np.array([
        result["gyro_x_highpass_rms"],
        result["gyro_y_highpass_rms"],
        result["gyro_z_highpass_rms"],
    ])

    result[
        "accel_vibration_rms_mean_axes"
    ] = float(
        np.mean(accel_vib)
    )

    result[
        "accel_vibration_rms_vector"
    ] = float(
        np.sqrt(
            np.sum(
                accel_vib ** 2
            )
        )
    )

    result[
        "gyro_vibration_rms_mean_axes"
    ] = float(
        np.mean(gyro_vib)
    )

    result[
        "gyro_vibration_rms_vector"
    ] = float(
        np.sqrt(
            np.sum(
                gyro_vib ** 2
            )
        )
    )

    # --------------------------------------------------------
    # Clipping
    # --------------------------------------------------------

    if "accelerometer_clipping" in window.columns:

        clip = clipping_counts(
            window[
                "accelerometer_clipping"
            ]
        )

        for key, value in clip.items():

            result[
                f"accel_{key}"
            ] = value

    else:

        for key in [
            "clip_x",
            "clip_y",
            "clip_z",
            "clip_any",
        ]:

            result[
                f"accel_{key}"
            ] = np.nan

    if "gyro_clipping" in window.columns:

        clip = clipping_counts(
            window["gyro_clipping"]
        )

        for key, value in clip.items():

            result[
                f"gyro_{key}"
            ] = value

    else:

        for key in [
            "clip_x",
            "clip_y",
            "clip_z",
            "clip_any",
        ]:

            result[
                f"gyro_{key}"
            ] = np.nan

    # --------------------------------------------------------
    # Calibration
    # --------------------------------------------------------

    accel_cal = (
        window["accel_calibration_count"]
        if "accel_calibration_count"
        in window.columns
        else None
    )

    gyro_cal = (
        window["gyro_calibration_count"]
        if "gyro_calibration_count"
        in window.columns
        else None
    )

    cal = calibration_info(
        accel_cal,
        gyro_cal
    )

    result.update(cal)

    return result


# ============================================================
# PROCESS ONE FLIGHT
# ============================================================

def process_flight(flight_id):

    input_file = (
        BASE_DIR
        / f"flight{flight_id}"
        / "PX4"
        / "CSV"
        / f"flight{flight_id}_sensor_combined_0.csv"
    )

    print("\n" + "=" * 72)
    print(
        f"FLIGHT {flight_id} - WINDOW FEATURE EXTRACTION"
    )
    print("=" * 72)

    print(
        f"Input:\n{input_file}"
    )

    if not input_file.exists():

        raise FileNotFoundError(
            f"File not found:\n{input_file}"
        )

    df = pd.read_csv(
        input_file
    )

    print(
        f"Rows    : {len(df):,}"
    )

    print(
        f"Columns : {len(df.columns)}"
    )

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    missing = [
        c for c in REQUIRED_COLS
        if c not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing columns in Flight {flight_id}: "
            f"{missing}"
        )

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    for col in REQUIRED_COLS:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Remove only rows with invalid required values
    #
    # No interpolation.
    # --------------------------------------------------------

    valid_mask = np.ones(
        len(df),
        dtype=bool
    )

    for col in REQUIRED_COLS:

        valid_mask &= np.isfinite(
            df[col].to_numpy(
                dtype=float
            )
        )

    invalid_rows = int(
        np.sum(~valid_mask)
    )

    if invalid_rows > 0:

        warnings.warn(
            f"Flight {flight_id}: "
            f"{invalid_rows} invalid rows removed. "
            f"No interpolation performed."
        )

        df = df.loc[
            valid_mask
        ].reset_index(drop=True)

    # --------------------------------------------------------
    # Global timestamp quality
    # --------------------------------------------------------

    global_tq = timestamp_quality(
        df["timestamp"].to_numpy(
            dtype=float
        )
    )

    print(
        f"Duration    : "
        f"{global_tq['duration_s']:.3f} s"
    )

    print(
        f"Median Fs   : "
        f"{global_tq['median_fs_hz']:.3f} Hz"
    )

    print(
        f"Jitter STD  : "
        f"{global_tq['jitter_std_percent']:.4f} %"
    )

    print(
        f"Gaps >2x    : "
        f"{global_tq['gaps_gt_2x']}"
    )

    print(
        f"Quality     : "
        f"{global_tq['quality']}"
    )

    # --------------------------------------------------------
    # Sampling frequency for filter
    #
    # Use median measured Fs.
    #
    # We do NOT resample.
    # --------------------------------------------------------

    fs = global_tq[
        "median_fs_hz"
    ]

    sos_accel = design_highpass(
        fs,
        VIBRATION_HIGHPASS_HZ,
        FILTER_ORDER
    )

    sos_gyro = design_highpass(
        fs,
        VIBRATION_HIGHPASS_HZ,
        FILTER_ORDER
    )

    # --------------------------------------------------------
    # Window indexing
    #
    # Windows are defined by timestamps, not row count.
    # --------------------------------------------------------

    timestamp = df[
        "timestamp"
    ].to_numpy(
        dtype=float
    )

    t0 = timestamp[0]

    elapsed_s = (
        timestamp - t0
    ) * 1e-6

    total_duration = elapsed_s[-1]

    n_windows = int(
        np.floor(
            total_duration
            / WINDOW_SEC
        )
    )

    print(
        f"30-s complete windows: "
        f"{n_windows}"
    )

    rows = []

    for window_id in range(
        n_windows
    ):

        start_s = (
            window_id
            * WINDOW_SEC
        )

        end_s = (
            start_s
            + WINDOW_SEC
        )

        mask = (
            (elapsed_s >= start_s)
            &
            (elapsed_s < end_s)
        )

        indices = np.where(
            mask
        )[0]

        if len(indices) < 100:

            continue

        result = extract_window_features(
            df=df,
            flight_id=flight_id,
            window_id=window_id,
            start_idx=indices[0],
            end_idx=indices[-1] + 1,
            sos_accel=sos_accel,
            sos_gyro=sos_gyro,
        )

        if result is not None:
            rows.append(result)

    feature_df = pd.DataFrame(
        rows
    )

    # --------------------------------------------------------
    # Add global metadata
    # --------------------------------------------------------

    feature_df[
        "global_duration_s"
    ] = global_tq["duration_s"]

    feature_df[
        "global_median_fs_hz"
    ] = global_tq["median_fs_hz"]

    feature_df[
        "global_jitter_std_percent"
    ] = global_tq[
        "jitter_std_percent"
    ]

    feature_df[
        "vibration_highpass_cutoff_hz"
    ] = VIBRATION_HIGHPASS_HZ

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        OUTPUT_DIR
        / f"flight{flight_id}_window_features.csv"
    )

    feature_df.to_csv(
        output_file,
        index=False
    )

    print(
        f"Saved: {output_file}"
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print(
        f"Extracted windows: "
        f"{len(feature_df)}"
    )

    if len(feature_df) > 0:

        print(
            "\nWindow timestamp quality:"
        )

        print(
            feature_df[
                "timestamp_quality"
            ].value_counts()
        )

        print(
            "\nMean high-frequency vibration:"
        )

        print(
            f"Accel: "
            f"{feature_df['accel_vibration_rms_mean_axes'].mean():.6f}"
        )

        print(
            f"Gyro : "
            f"{feature_df['gyro_vibration_rms_mean_axes'].mean():.6f}"
        )

    return feature_df, global_tq


# ============================================================
# MAIN
# ============================================================

def main():

    all_features = []

    global_quality = []

    for flight_id in FLIGHTS:

        features, quality = process_flight(
            flight_id
        )

        all_features.append(
            features
        )

        quality_row = {
            "flight": flight_id,
            **quality,
        }

        global_quality.append(
            quality_row
        )

    # --------------------------------------------------------
    # Combined dataset
    # --------------------------------------------------------

    combined = pd.concat(
        all_features,
        ignore_index=True
    )

    combined_file = (
        OUTPUT_DIR
        / "window_features_F2_F3_F5.csv"
    )

    combined.to_csv(
        combined_file,
        index=False
    )

    # --------------------------------------------------------
    # Quality table
    # --------------------------------------------------------

    quality_df = pd.DataFrame(
        global_quality
    )

    quality_file = (
        OUTPUT_DIR
        / "flight2_3_5_quality_summary.csv"
    )

    quality_df.to_csv(
        quality_file,
        index=False
    )

    # --------------------------------------------------------
    # Excel report
    # --------------------------------------------------------

    excel_file = (
        OUTPUT_DIR
        / "flight2_3_5_data_quality.xlsx"
    )

    with pd.ExcelWriter(
        excel_file,
        engine="openpyxl"
    ) as writer:

        quality_df.to_excel(
            writer,
            sheet_name="Global_Quality",
            index=False
        )

        combined.to_excel(
            writer,
            sheet_name="Window_Features",
            index=False
        )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print("\n" + "=" * 72)
    print("COMBINED DATASET")
    print("=" * 72)

    print(
        f"Total windows: "
        f"{len(combined)}"
    )

    print(
        "\nWindows by flight:"
    )

    print(
        combined[
            "flight"
        ].value_counts()
        .sort_index()
    )

    print(
        "\nTimestamp quality:"
    )

    print(
        combined[
            "timestamp_quality"
        ].value_counts()
    )

    print(
        "\nSaved combined dataset:"
    )

    print(combined_file)

    print(
        "\nSaved quality summary:"
    )

    print(quality_file)

    print(
        "\nSaved Excel report:"
    )

    print(excel_file)

    print("\nDone.")


if __name__ == "__main__":
    main()