"""
================================================================================
WINDOW FEATURE EXTRACTOR - Corrected Version
================================================================================
This module extracts all window-level features from IMU data including:
- Time information
- IMU variability (mean, std, rms, mad)
- Corrected vibration features (band-pass filtered)
- Quality flags (clipping, calibration)

Author: Research Team
Date: 2024
================================================================================
"""

import numpy as np
import pandas as pd
import os
import sys
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

# Import the corrected vibration extractor
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from vibration_extractor import VibrationExtractor, VibrationValidator


class WindowFeatureExtractor:
    """
    Extract comprehensive window-level features from IMU data.
    
    Parameters:
    -----------
    data_path : str
        Path to the sensor_combined CSV file
    flight_id : str
        Flight identifier (e.g., 'flight2')
    window_sec : float
        Window duration in seconds (default: 30.0)
    fs_target : float
        Expected sampling frequency (for validation)
    """
    
    def __init__(self, data_path, flight_id, window_sec=30.0, fs_target=200.0):
        """Initialize the window feature extractor."""
        self.data_path = data_path
        self.flight_id = flight_id
        self.window_sec = window_sec
        self.fs_target = fs_target
        
        # Data containers
        self.df = None
        self.time = None
        self.accel = None
        self.gyro = None
        self.fs = None
        
        # Quality flags
        self.clipping_flag = None
        self.calibration_change = None
        
        # Results
        self.windows = None
        
        print("=" * 70)
        print(f"WINDOW FEATURE EXTRACTOR - {flight_id.upper()}")
        print("=" * 70)
        print(f"Input     : {data_path}")
        print(f"Window    : {window_sec} s")
        print(f"Target Fs : {fs_target} Hz")
        print("=" * 70)
    
    def load_data(self):
        """Load and parse the sensor_combined CSV file."""
        print("\n📂 Loading data...")
        
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        self.df = pd.read_csv(self.data_path)
        
        print(f"   Rows    : {len(self.df):,}")
        print(f"   Columns : {len(self.df.columns)}")
        print(f"   Columns : {self.df.columns.tolist()}")
        
        # Extract time (convert from microseconds to seconds)
        if 'timestamp' in self.df.columns:
            self.time = (self.df['timestamp'] - self.df['timestamp'].iloc[0]) / 1e6
        else:
            raise ValueError("No 'timestamp' column found")
        
        # Extract accelerometer data
        accel_cols = self._find_columns(
            candidates=[
                ['accelerometer_m_s2[0]', 'accelerometer_m_s2[1]', 'accelerometer_m_s2[2]'],
                ['accelerometer_m_s2_0', 'accelerometer_m_s2_1', 'accelerometer_m_s2_2'],
                ['accel_x', 'accel_y', 'accel_z'],
                ['ax', 'ay', 'az']
            ]
        )
        
        if accel_cols is None:
            raise ValueError("Could not find accelerometer columns")
        
        self.accel = self.df[accel_cols].values
        print(f"\n   ✅ Accelerometer columns: {accel_cols}")
        
        # Extract gyroscope data
        gyro_cols = self._find_columns(
            candidates=[
                ['gyro_rad[0]', 'gyro_rad[1]', 'gyro_rad[2]'],
                ['gyro_rad_0', 'gyro_rad_1', 'gyro_rad_2'],
                ['gyro_x', 'gyro_y', 'gyro_z'],
                ['gx', 'gy', 'gz']
            ]
        )
        
        if gyro_cols is None:
            raise ValueError("Could not find gyroscope columns")
        
        self.gyro = self.df[gyro_cols].values
        print(f"   ✅ Gyroscope columns: {gyro_cols}")
        
        # Extract quality flags
        self._extract_quality_flags()
        
        # Estimate sampling frequency
        dt = np.diff(self.time.values)
        self.fs = 1.0 / np.median(dt)
        
        print(f"\n   ✅ Estimated Fs: {self.fs:.2f} Hz")
        print(f"   ✅ Duration    : {self.time.iloc[-1]:.2f} s")
        print(f"   ✅ Samples     : {len(self.df):,}")
        
        return self
    
    def _find_columns(self, candidates):
        """Find the first matching set of columns."""
        for cols in candidates:
            if all(col in self.df.columns for col in cols):
                return cols
        return None
    
    def _extract_quality_flags(self):
        """Extract clipping and calibration quality flags."""
        # Clipping flags
        clip_cols = []
        for col in self.df.columns:
            if 'clipping' in col.lower():
                clip_cols.append(col)
        
        if clip_cols:
            self.clipping_flag = self.df[clip_cols].any(axis=1).values
            n_clips = self.clipping_flag.sum()
            print(f"   ⚠️  Clipping events: {n_clips}")
        else:
            self.clipping_flag = np.zeros(len(self.df), dtype=bool)
            print(f"   ✅ No clipping columns found")
        
        # Calibration counters
        cal_cols = []
        for col in self.df.columns:
            if 'calibration_count' in col.lower():
                cal_cols.append(col)
        
        if cal_cols:
            cal_data = self.df[cal_cols].values
            # Detect changes
            cal_changes = np.any(np.diff(cal_data, axis=0) != 0, axis=1)
            self.calibration_change = np.concatenate([[False], cal_changes])
            n_changes = self.calibration_change.sum()
            print(f"   ⚠️  Calibration changes: {n_changes}")
        else:
            self.calibration_change = np.zeros(len(self.df), dtype=bool)
            print(f"   ✅ No calibration counter columns found")
    
    def assess_timestamp_quality(self):
        """Assess timestamp quality."""
        print("\n📊 Assessing timestamp quality...")
        
        dt = np.diff(self.time.values)
        median_dt = np.median(dt)
        mean_dt = np.mean(dt)
        std_dt = np.std(dt)
        
        # Jitter as percentage
        jitter_std = (std_dt / mean_dt) * 100
        
        # MAD-based jitter (more robust)
        mad = np.median(np.abs(dt - median_dt))
        jitter_mad = (mad / median_dt) * 100
        
        # Gaps
        gaps_2x = np.sum(dt > 2 * median_dt)
        gaps_5x = np.sum(dt > 5 * median_dt)
        
        # Largest gap
        largest_gap = np.max(dt)
        
        quality = "GOOD"
        if jitter_std > 5 or gaps_5x > 0:
            quality = "WARNING"
        if jitter_std > 20 or gaps_5x > 10:
            quality = "BAD"
        
        metrics = {
            'duration': self.time.iloc[-1],
            'samples': len(self.df),
            'median_fs': 1.0 / median_dt,
            'mean_fs': 1.0 / mean_dt,
            'jitter_std_pct': jitter_std,
            'jitter_mad_pct': jitter_mad,
            'gaps_2x': int(gaps_2x),
            'gaps_5x': int(gaps_5x),
            'largest_gap_s': largest_gap,
            'quality': quality
        }
        
        print(f"   Duration        : {metrics['duration']:.2f} s")
        print(f"   Samples         : {metrics['samples']:,}")
        print(f"   Median Fs       : {metrics['median_fs']:.2f} Hz")
        print(f"   Mean Fs         : {metrics['mean_fs']:.2f} Hz")
        print(f"   Jitter STD      : {metrics['jitter_std_pct']:.4f} %")
        print(f"   Jitter MAD      : {metrics['jitter_mad_pct']:.4f} %")
        print(f"   Gaps > 2×dt     : {metrics['gaps_2x']}")
        print(f"   Gaps > 5×dt     : {metrics['gaps_5x']}")
        print(f"   Largest gap     : {metrics['largest_gap_s']*1000:.3f} ms")
        print(f"   Quality         : {metrics['quality']}")
        
        self.timestamp_quality = metrics
        return metrics
    
    def extract_window_features(self):
        """Extract all window-level features."""
        print("\n📊 Extracting window features...")
        
        # Determine window parameters
        window_samples = int(self.window_sec * self.fs)
        n_windows = len(self.df) // window_samples
        
        print(f"   Window samples  : {window_samples:,}")
        print(f"   Number of windows: {n_windows}")
        
        # Initialize feature storage
        features = []
        
        # Process vibration with corrected extractor
        print("\n   🔄 Extracting vibration features...")
        vib_extractor = VibrationExtractor(
            fs=self.fs,
            low_cut=5.0,
            high_cut=50.0,
            order=6
        )
        
        # Vibration RMS per window
        vib_accel_rms, vib_accel_times = vib_extractor.extract_windowed_rms(
            self.accel, window_sec=self.window_sec
        )
        vib_gyro_rms, vib_gyro_times = vib_extractor.extract_windowed_rms(
            self.gyro, window_sec=self.window_sec
        )
        
        # Ensure consistent number of windows
        n_vib = min(len(vib_accel_rms), n_windows)
        
        # Extract features per window
        for i in range(n_vib):
            start = i * window_samples
            end = start + window_samples
            
            # Time
            window_time = self.time.iloc[start]
            
            # Accelerometer features
            accel_win = self.accel[start:end, :]
            accel_feats = self._compute_statistics(accel_win, 'accel')
            
            # Gyroscope features
            gyro_win = self.gyro[start:end, :]
            gyro_feats = self._compute_statistics(gyro_win, 'gyro')
            
            # Quality flags
            clip_in_window = self.clipping_flag[start:end].any()
            cal_in_window = self.calibration_change[start:end].any()
            
            # Vibration features
            vib_accel = vib_accel_rms[i, :]
            vib_gyro = vib_gyro_rms[i, :]
            
            vib_accel_total = np.sqrt(np.sum(vib_accel ** 2))
            vib_gyro_total = np.sqrt(np.sum(vib_gyro ** 2))
            
            # Build feature row
            row = {
                'flight': self.flight_id,
                'window_id': i,
                'time_s': window_time,
                'timestamp_quality': 'GOOD',
                'clipping_flag': bool(clip_in_window),
                'calibration_change': bool(cal_in_window),
                'vib_accel_x': vib_accel[0],
                'vib_accel_y': vib_accel[1],
                'vib_accel_z': vib_accel[2],
                'vib_gyro_x': vib_gyro[0],
                'vib_gyro_y': vib_gyro[1],
                'vib_gyro_z': vib_gyro[2],
                'vib_accel_total': vib_accel_total,
                'vib_gyro_total': vib_gyro_total
            }
            
            # Add IMU statistics
            row.update(accel_feats)
            row.update(gyro_feats)
            
            features.append(row)
        
        self.windows = pd.DataFrame(features)
        
        print(f"\n   ✅ Extracted {len(self.windows)} windows")
        
        return self.windows
    
    def _compute_statistics(self, data, prefix):
        """Compute statistical features for a window."""
        axes = ['x', 'y', 'z']
        feats = {}
        
        for i, axis in enumerate(axes):
            axis_data = data[:, i]
            feats[f'{prefix}_{axis}_mean'] = np.mean(axis_data)
            feats[f'{prefix}_{axis}_std'] = np.std(axis_data)
            feats[f'{prefix}_{axis}_rms'] = np.sqrt(np.mean(axis_data ** 2))
            feats[f'{prefix}_{axis}_mad'] = np.median(
                np.abs(axis_data - np.median(axis_data))
            )
            feats[f'{prefix}_{axis}_min'] = np.min(axis_data)
            feats[f'{prefix}_{axis}_max'] = np.max(axis_data)
        
        return feats
    
    def validate_vibration(self):
        """Validate extracted vibration features."""
        if self.windows is None:
            raise ValueError("No windows extracted yet")
        
        print("\n" + "=" * 70)
        print("VIBRATION VALIDATION")
        print("=" * 70)
        
        # Global STD
        global_std_accel = np.std(self.accel, axis=0)
        global_std_gyro = np.std(self.gyro, axis=0)
        
        # Validator
        validator = VibrationValidator(max_ratio=0.20)
        
        # Validate accelerometer
        print("\n📊 Accelerometer Vibration:")
        accel_vib = self.windows[['vib_accel_x', 'vib_accel_y', 'vib_accel_z']].values
        accel_results = validator.validate(accel_vib, global_std_accel)
        
        # Validate gyroscope
        print("\n📊 Gyroscope Vibration:")
        gyro_vib = self.windows[['vib_gyro_x', 'vib_gyro_y', 'vib_gyro_z']].values
        gyro_results = validator.validate(gyro_vib, global_std_gyro)
        
        return {
            'accel': accel_results,
            'gyro': gyro_results
        }
    
    def save_features(self, output_path):
        """Save extracted features to CSV."""
        if self.windows is None:
            raise ValueError("No windows extracted yet")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.windows.to_csv(output_path, index=False)
        
        print(f"\n💾 Features saved to: {output_path}")
        print(f"   Rows    : {len(self.windows)}")
        print(f"   Columns : {len(self.windows.columns)}")
        
        return self.windows


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_pipeline_for_flight(flight_id, data_path, output_dir):
    """Run the complete pipeline for a single flight."""
    print("\n" + "=" * 70)
    print(f"PROCESSING {flight_id.upper()}")
    print("=" * 70)
    
    # Initialize extractor
    extractor = WindowFeatureExtractor(
        data_path=data_path,
        flight_id=flight_id,
        window_sec=30.0,
        fs_target=200.0
    )
    
    # Load data
    extractor.load_data()
    
    # Assess quality
    extractor.assess_timestamp_quality()
    
    # Extract features
    windows = extractor.extract_window_features()
    
    # Validate vibration
    validation = extractor.validate_vibration()
    
    # Save
    output_path = os.path.join(output_dir, f'{flight_id}_window_features_v2.csv')
    extractor.save_features(output_path)
    
    return extractor, windows, validation


def run_full_pipeline():
    """Run the complete pipeline for all flights."""
    print("=" * 70)
    print("FULL WINDOW FEATURE EXTRACTION PIPELINE (CORRECTED)")
    print("=" * 70)
    
    # Configuration
    base_dir = "/home/fatima/Research/validation"
    output_dir = os.path.join(
        base_dir,
        "imu_degradation_results",
        "window_feature_analysis_v2"
    )
    os.makedirs(output_dir, exist_ok=True)
    
    # Flight configuration
    flights = {
        'flight2': os.path.join(
            base_dir, 'flight2/PX4/CSV/flight2_sensor_combined_0.csv'
        ),
        'flight3': os.path.join(
            base_dir, 'flight3/PX4/CSV/flight3_sensor_combined_0.csv'
        ),
        'flight5': os.path.join(
            base_dir, 'flight5/PX4/CSV/flight5_sensor_combined_0.csv'
        )
    }
    
    # Process each flight
    all_results = {}
    all_windows = []
    
    for flight_id, data_path in flights.items():
        try:
            extractor, windows, validation = run_pipeline_for_flight(
                flight_id, data_path, output_dir
            )
            
            all_results[flight_id] = {
                'extractor': extractor,
                'windows': windows,
                'validation': validation
            }
            
            all_windows.append(windows)
            
        except Exception as e:
            print(f"\n❌ Error processing {flight_id}: {e}")
            import traceback
            traceback.print_exc()
    
    # Combine all windows
    if all_windows:
        combined = pd.concat(all_windows, ignore_index=True)
        
        combined_path = os.path.join(output_dir, 'window_features_F2_F3_F5_v2.csv')
        combined.to_csv(combined_path, index=False)
        
        print("\n" + "=" * 70)
        print("COMBINED DATASET")
        print("=" * 70)
        print(f"Total windows: {len(combined)}")
        print("\nWindows by flight:")
        print(combined['flight'].value_counts().sort_index())
        print(f"\nSaved: {combined_path}")
        
        # Quality summary
        quality_summary = combined.groupby('flight').agg({
            'window_id': 'count',
            'clipping_flag': 'sum',
            'calibration_change': 'sum',
            'vib_accel_total': 'mean',
            'vib_gyro_total': 'mean'
        }).rename(columns={'window_id': 'n_windows'})
        
        quality_path = os.path.join(output_dir, 'flight_quality_summary_v2.csv')
        quality_summary.to_csv(quality_path)
        
        print("\n" + "=" * 70)
        print("QUALITY SUMMARY")
        print("=" * 70)
        print(quality_summary.to_string())
        print(f"\nSaved: {quality_path}")
    
    print("\n" + "=" * 70)
    print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    
    return all_results


if __name__ == "__main__":
    results = run_full_pipeline()