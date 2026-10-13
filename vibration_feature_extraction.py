#!/usr/bin/env python3
"""
================================================================================
STEP 7: VIBRATION FEATURE EXTRACTION
================================================================================
This script extracts vibration features from IMU data using the selected
frequency bands.

Key principles:
1. No target leakage: vibration features are independent of noise metrics
2. Use band-pass filter for each selected band
3. Compute windowed RMS for each band and axis
4. Preserve timestamp quality information

Author: Research Team
Date: 2024
================================================================================
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import signal

warnings.filterwarnings('ignore')


# ============================================================
# CONFIGURATION
# ============================================================

FLIGHTS = {
    "flight2": Path(
        "/home/fatima/Research/validation/flight2/PX4/CSV/"
        "flight2_sensor_combined_0.csv"
    ),
    "flight3": Path(
        "/home/fatima/Research/validation/flight3/PX4/CSV/"
        "flight3_sensor_combined_0.csv"
    ),
    "flight5": Path(
        "/home/fatima/Research/validation/flight5/PX4/CSV/"
        "flight5_sensor_combined_0.csv"
    ),
}

OUTPUT_DIR = Path(
    "/home/fatima/Research/validation/"
    "imu_degradation_results/vibration_features"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Vibration bands (from STEP 6)
VIBRATION_BANDS = {
    "motion": (0.1, 5.0),
    "vibration": (5.0, 50.0),
    "high_freq": (50.0, 100.0),
}

# Window settings
WINDOW_SECONDS = 30.0

# Sampling frequency
FS = 200.0

# Columns
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
# BAND-PASS FILTER
# ============================================================

def design_bandpass_filter(fs, low, high, order=6):
    """Design a band-pass filter using SOS."""
    nyq = fs / 2.0
    
    # Ensure valid frequencies
    low_norm = max(low / nyq, 1e-6)
    high_norm = min(high / nyq, 0.999)
    
    if low_norm >= high_norm:
        raise ValueError(f"Invalid band: {low}-{high} Hz")
    
    sos = signal.butter(
        N=order,
        Wn=[low_norm, high_norm],
        btype='band',
        output='sos'
    )
    
    return sos


def apply_bandpass(data, sos):
    """Apply band-pass filter with zero-phase."""
    # Handle NaN
    data_clean = data.copy()
    mask = np.isfinite(data_clean)
    
    if mask.sum() < 100:
        return np.zeros_like(data_clean)
    
    # Interpolate NaN if any
    if not mask.all():
        data_clean = np.interp(
            np.arange(len(data_clean)),
            np.arange(len(data_clean))[mask],
            data_clean[mask]
        )
    
    # Apply filter
    filtered = signal.sosfiltfilt(sos, data_clean)
    
    return filtered


# ============================================================
# WINDOW EXTRACTION
# ============================================================

def get_complete_windows(df, window_seconds=30.0):
    """Get complete timestamp-based windows."""
    t = df["timestamp"].to_numpy(dtype=np.float64) / 1e6
    
    start_time = t[0]
    end_time = t[-1]
    
    n_complete = int(np.floor((end_time - start_time) / window_seconds))
    
    windows = []
    
    for i in range(n_complete):
        window_start = start_time + i * window_seconds
        window_end = window_start + window_seconds
        
        mask = (t >= window_start) & (t < window_end)
        indices = np.flatnonzero(mask)
        
        if len(indices) == 0:
            continue
        
        actual_duration = t[indices[-1]] - t[indices[0]]
        
        if actual_duration < 0.98 * window_seconds:
            continue
        
        windows.append({
            "window_id": i,
            "start_time_s": window_start,
            "end_time_s": window_end,
            "indices": indices,
            "n_samples": len(indices),
        })
    
    return windows


# ============================================================
# TIMESTAMP QUALITY
# ============================================================

def estimate_sampling_quality(timestamp_us):
    """Estimate sampling frequency and quality."""
    t = np.asarray(timestamp_us, dtype=np.float64) / 1e6
    dt = np.diff(t)
    valid_dt = dt[np.isfinite(dt) & (dt > 0)]
    
    if len(valid_dt) == 0:
        return {
            "fs_median": np.nan,
            "jitter_std_percent": np.nan,
            "gaps_gt_2x": np.nan,
        }
    
    median_dt = np.median(valid_dt)
    fs_median = 1.0 / median_dt
    
    jitter_percent = (valid_dt - median_dt) / median_dt * 100.0
    jitter_std_percent = np.std(jitter_percent)
    
    gaps_gt_2x = int(np.sum(valid_dt > 2.0 * median_dt))
    
    return {
        "fs_median": fs_median,
        "jitter_std_percent": jitter_std_percent,
        "gaps_gt_2x": gaps_gt_2x,
    }


def is_window_good(timestamp_us, max_jitter=5.0, max_gaps=0):
    """Check if window is acceptable."""
    quality = estimate_sampling_quality(timestamp_us)
    
    if not np.isfinite(quality["fs_median"]):
        return False, quality
    
    if not (180.0 <= quality["fs_median"] <= 220.0):
        return False, quality
    
    if quality["jitter_std_percent"] > max_jitter:
        return False, quality
    
    if quality["gaps_gt_2x"] > max_gaps:
        return False, quality
    
    return True, quality


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_vibration_features(flight_name, csv_path):
    """
    Extract vibration features for a single flight.
    
    Features per window:
    - Motion RMS (0.1-5 Hz)
    - Vibration RMS (5-50 Hz)
    - High-freq RMS (50-100 Hz)
    For each axis (X, Y, Z) and each sensor (accel, gyro)
    """
    print("\n" + "=" * 70)
    print(f"EXTRACTING VIBRATION FEATURES - {flight_name.upper()}")
    print("=" * 70)
    
    # Load data
    if not csv_path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Keep only required columns
    required = ["timestamp"] + ACCEL_COLS + GYRO_COLS
    df = df[required].copy()
    df = df[np.isfinite(df["timestamp"])].copy()
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    print(f"Rows: {len(df):,}")
    
    # Get windows
    windows = get_complete_windows(df, WINDOW_SECONDS)
    print(f"Complete 30-s windows: {len(windows)}")
    
    # Design filters for each band
    filters = {}
    for band_name, (low, high) in VIBRATION_BANDS.items():
        filters[band_name] = design_bandpass_filter(FS, low, high)
    
    print(f"Bands: {list(VIBRATION_BANDS.keys())}")
    
    # Extract features per window
    features = []
    
    for window in windows:
        indices = window["indices"]
        win = df.iloc[indices]
        timestamp = win["timestamp"].to_numpy()
        
        # Check timestamp quality
        good, quality = is_window_good(timestamp)
        
        if not good:
            print(f"   ⚠️  Window {window['window_id']} skipped (quality)")
            continue
        
        fs = quality["fs_median"]
        
        # Base metadata
        row = {
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
        for i, (col, axis) in enumerate(zip(ACCEL_COLS, ACCEL_NAMES)):
            data = win[col].to_numpy(dtype=np.float64)
            
            # Remove DC
            data = data - np.mean(data)
            
            # Apply filters
            for band_name, sos in filters.items():
                filtered = apply_bandpass(data, sos)
                rms = np.sqrt(np.mean(filtered ** 2))
                row[f"{axis}_{band_name}_rms"] = rms
            
            # Total RMS (raw)
            row[f"{axis}_total_rms"] = np.sqrt(np.mean(data ** 2))
            row[f"{axis}_std"] = np.std(data)
        
        # Process gyroscope
        for i, (col, axis) in enumerate(zip(GYRO_COLS, GYRO_NAMES)):
            data = win[col].to_numpy(dtype=np.float64)
            
            # Remove DC
            data = data - np.mean(data)
            
            # Apply filters
            for band_name, sos in filters.items():
                filtered = apply_bandpass(data, sos)
                rms = np.sqrt(np.mean(filtered ** 2))
                row[f"{axis}_{band_name}_rms"] = rms
            
            # Total RMS
            row[f"{axis}_total_rms"] = np.sqrt(np.mean(data ** 2))
            row[f"{axis}_std"] = np.std(data)
        
        # Compute combined vibration metrics
        # Accelerometer vibration (5-50 Hz)
        accel_vib = [
            row[f"accel_{ax}_vibration_rms"] 
            for ax in ["x", "y", "z"]
        ]
        row["accel_vibration_total"] = np.sqrt(sum(v**2 for v in accel_vib))
        
        # Gyroscope vibration (5-50 Hz)
        gyro_vib = [
            row[f"gyro_{ax}_vibration_rms"] 
            for ax in ["x", "y", "z"]
        ]
        row["gyro_vibration_total"] = np.sqrt(sum(v**2 for v in gyro_vib))
        
        # High-frequency (50-100 Hz)
        accel_hf = [
            row[f"accel_{ax}_high_freq_rms"] 
            for ax in ["x", "y", "z"]
        ]
        row["accel_high_freq_total"] = np.sqrt(sum(v**2 for v in accel_hf))
        
        gyro_hf = [
            row[f"gyro_{ax}_high_freq_rms"] 
            for ax in ["x", "y", "z"]
        ]
        row["gyro_high_freq_total"] = np.sqrt(sum(v**2 for v in gyro_hf))
        
        # Motion (0.1-5 Hz)
        accel_motion = [
            row[f"accel_{ax}_motion_rms"] 
            for ax in ["x", "y", "z"]
        ]
        row["accel_motion_total"] = np.sqrt(sum(v**2 for v in accel_motion))
        
        gyro_motion = [
            row[f"gyro_{ax}_motion_rms"] 
            for ax in ["x", "y", "z"]
        ]
        row["gyro_motion_total"] = np.sqrt(sum(v**2 for v in gyro_motion))
        
        features.append(row)
    
    features_df = pd.DataFrame(features)
    
    print(f"✅ Extracted {len(features_df)} windows")
    
    return features_df


# ============================================================
# VALIDATION
# ============================================================

def validate_vibration_features(df):
    """Validate extracted vibration features."""
    print("\n" + "=" * 70)
    print("VIBRATION FEATURE VALIDATION")
    print("=" * 70)
    
    # Check ranges
    print("\n📊 Feature Ranges:")
    print("-" * 50)
    
    for col in df.columns:
        if 'vibration_total' in col or 'high_freq_total' in col:
            values = df[col].dropna()
            if len(values) > 0:
                print(f"   {col}:")
                print(f"      Min:  {values.min():.4e}")
                print(f"      Max:  {values.max():.4e}")
                print(f"      Mean: {values.mean():.4e}")
    
    # Check for anomalies
    print("\n📊 Anomaly Check:")
    print("-" * 50)
    
    # Negative or zero values
    for col in df.columns:
        if 'rms' in col:
            n_zero = (df[col] == 0).sum()
            n_neg = (df[col] < 0).sum()
            if n_zero > 0 or n_neg > 0:
                print(f"   ⚠️  {col}: {n_zero} zeros, {n_neg} negatives")
    
    # Vibration vs Motion ratio
    print("\n📊 Vibration vs Motion Ratio:")
    print("-" * 50)
    
    if 'accel_vibration_total' in df.columns and 'accel_motion_total' in df.columns:
        ratio = df['accel_vibration_total'] / (df['accel_motion_total'] + 1e-10)
        print(f"   Accel: mean={ratio.mean():.4f}, "
              f"min={ratio.min():.4f}, max={ratio.max():.4f}")
    
    if 'gyro_vibration_total' in df.columns and 'gyro_motion_total' in df.columns:
        ratio = df['gyro_vibration_total'] / (df['gyro_motion_total'] + 1e-10)
        print(f"   Gyro:  mean={ratio.mean():.4f}, "
              f"min={ratio.min():.4f}, max={ratio.max():.4f}")
    
    print("\n✅ Validation complete")


# ============================================================
# MAIN
# ============================================================

def main():
    """Main execution."""
    print("=" * 70)
    print("STEP 7: VIBRATION FEATURE EXTRACTION")
    print("=" * 70)
    
    all_features = []
    
    for flight_name, csv_path in FLIGHTS.items():
        try:
            features_df = extract_vibration_features(flight_name, csv_path)
            
            # Save individual flight
            output_path = OUTPUT_DIR / f"{flight_name}_vibration_features.csv"
            features_df.to_csv(output_path, index=False)
            print(f"✅ Saved: {output_path}")
            
            all_features.append(features_df)
        
        except Exception as e:
            print(f"❌ Error processing {flight_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Combine all flights
    if all_features:
        combined = pd.concat(all_features, ignore_index=True)
        
        combined_path = OUTPUT_DIR / "combined_vibration_features.csv"
        combined.to_csv(combined_path, index=False)
        
        print(f"\n✅ Combined dataset: {combined_path}")
        print(f"   Total windows: {len(combined)}")
        print(f"   Columns: {len(combined.columns)}")
        
        # Validate
        validate_vibration_features(combined)
        
        # Print summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        
        print("\n📊 Windows by flight:")
        print(combined['flight'].value_counts().sort_index())
        
        print("\n📊 Vibration means by flight:")
        summary = combined.groupby('flight').agg({
            'accel_vibration_total': 'mean',
            'gyro_vibration_total': 'mean',
            'accel_high_freq_total': 'mean',
            'gyro_high_freq_total': 'mean',
            'accel_motion_total': 'mean',
            'gyro_motion_total': 'mean',
        }).round(4)
        
        print(summary.to_string())
    
    print("\n" + "=" * 70)
    print("✅ STEP 7 COMPLETED")
    print("=" * 70)
    
    return combined if all_features else None


if __name__ == "__main__":
    combined = main()