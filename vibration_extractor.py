"""
================================================================================
VIBRATION EXTRACTOR - Corrected Version
================================================================================
This module extracts vibration features from IMU data using proper band-pass
filtering to separate high-frequency vibration from low-frequency motion.

Author: Research Team
Date: 2024
================================================================================
"""

import numpy as np
import pandas as pd
from scipy import signal
import warnings
warnings.filterwarnings('ignore')


class VibrationExtractor:
    """
    Corrected vibration extraction using band-pass filter.
    
    This class addresses the critical issue where high-pass filtering alone
    was insufficient to remove low-frequency motion content, resulting in
    physically implausible vibration values for dynamic flights.
    
    Parameters:
    -----------
    fs : float
        Sampling frequency (Hz)
    low_cut : float
        Low cutoff frequency for vibration band (Hz)
    high_cut : float
        High cutoff frequency for vibration band (Hz)
    order : int
        Filter order (higher = steeper rolloff)
    """
    
    def __init__(self, fs=200.0, low_cut=5.0, high_cut=50.0, order=6):
        """Initialize the vibration extractor with proper filter parameters."""
        self.fs = fs
        self.low_cut = low_cut
        self.high_cut = high_cut
        self.order = order
        
        # Validate cutoff frequencies
        nyq = fs / 2.0
        if low_cut >= high_cut:
            raise ValueError(f"low_cut ({low_cut}) must be < high_cut ({high_cut})")
        if high_cut >= nyq:
            raise ValueError(f"high_cut ({high_cut}) must be < Nyquist ({nyq})")
        
        # Design band-pass filter using SOS (numerically stable)
        self.sos = signal.butter(
            N=order,
            Wn=[low_cut / nyq, high_cut / nyq],
            btype='band',
            output='sos'
        )
        
        print("=" * 70)
        print("VIBRATION EXTRACTOR INITIALIZED")
        print("=" * 70)
        print(f"Sampling Frequency : {fs:.2f} Hz")
        print(f"Vibration Band     : {low_cut:.2f} - {high_cut:.2f} Hz")
        print(f"Filter Order       : {order}")
        print(f"Nyquist Frequency  : {nyq:.2f} Hz")
        print("=" * 70)
    
    def apply_filter(self, data):
        """
        Apply band-pass filter to data.
        
        Parameters:
        -----------
        data : np.ndarray
            Input data (N, 3) - 3 axes
        
        Returns:
        --------
        filtered : np.ndarray
            Band-pass filtered data (N, 3)
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        n_axes = data.shape[1]
        filtered = np.zeros_like(data)
        
        for i in range(n_axes):
            # Check for NaN or Inf
            if np.any(~np.isfinite(data[:, i])):
                # Replace non-finite values with interpolation
                mask = np.isfinite(data[:, i])
                if mask.sum() < 10:
                    filtered[:, i] = 0.0
                    continue
                data[:, i] = np.interp(
                    np.arange(len(data)),
                    np.arange(len(data))[mask],
                    data[mask, i]
                )
            
            # Apply zero-phase filtering
            filtered[:, i] = signal.sosfiltfilt(self.sos, data[:, i])
        
        return filtered
    
    def extract_windowed_rms(self, data, window_sec=30.0, overlap=0.0):
        """
        Extract windowed RMS of vibration signal.
        
        Parameters:
        -----------
        data : np.ndarray
            Input data (N, 3)
        window_sec : float
            Window duration in seconds
        overlap : float
            Overlap fraction (0.0 to 1.0)
        
        Returns:
        --------
        vibration_rms : np.ndarray
            Windowed vibration RMS (n_windows, 3)
        window_times : np.ndarray
            Start time of each window (seconds)
        """
        # Apply band-pass filter
        filtered = self.apply_filter(data)
        
        # Window parameters
        window_samples = int(window_sec * self.fs)
        step_samples = int(window_samples * (1 - overlap))
        
        if len(data) < window_samples:
            raise ValueError(
                f"Data length ({len(data)}) < window size ({window_samples})"
            )
        
        # Compute number of windows
        n_windows = (len(data) - window_samples) // step_samples + 1
        
        # Extract windowed RMS
        vibration_rms = np.zeros((n_windows, filtered.shape[1]))
        window_times = np.zeros(n_windows)
        
        for i in range(n_windows):
            start = i * step_samples
            end = start + window_samples
            
            # RMS of each axis
            vibration_rms[i, :] = np.sqrt(
                np.mean(filtered[start:end, :] ** 2, axis=0)
            )
            
            # Window start time
            window_times[i] = start / self.fs
        
        return vibration_rms, window_times
    
    def verify_extraction(self, data, verbose=True):
        """
        Verify vibration extraction quality.
        
        Parameters:
        -----------
        data : np.ndarray
            Input data (N, 3)
        verbose : bool
            Print verification results
        
        Returns:
        --------
        dict
            Verification metrics
        """
        metrics = {}
        
        for i, axis in enumerate(['X', 'Y', 'Z']):
            # Compute PSD
            freqs, psd = signal.welch(
                data[:, i],
                fs=self.fs,
                nperseg=min(2048, len(data))
            )
            
            # Vibration band mask
            vib_band = (freqs >= self.low_cut) & (freqs <= self.high_cut)
            
            # Energy in vibration band
            vib_energy = np.trapz(psd[vib_band], freqs[vib_band])
            total_energy = np.trapz(psd, freqs)
            
            # Vibration fraction
            vib_fraction = vib_energy / total_energy if total_energy > 0 else 0
            
            # Signal statistics
            global_std = np.std(data[:, i])
            
            metrics[f'axis_{axis}'] = {
                'global_std': global_std,
                'vibration_fraction': vib_fraction,
                'vibration_energy': vib_energy,
                'total_energy': total_energy
            }
            
            if verbose:
                print(f"\n  Axis {axis}:")
                print(f"    Global STD          : {global_std:.4f}")
                print(f"    Vibration Fraction  : {vib_fraction:.4f}")
                print(f"    Vibration Energy    : {vib_energy:.4e}")
                print(f"    Total Energy        : {total_energy:.4e}")
        
        # Overall assessment
        mean_fraction = np.mean([
            metrics[f'axis_{ax}']['vibration_fraction']
            for ax in ['X', 'Y', 'Z']
        ])
        
        metrics['mean_vibration_fraction'] = mean_fraction
        
        if verbose:
            print(f"\n  Mean Vibration Fraction: {mean_fraction:.4f}")
            
            if 0.05 <= mean_fraction <= 0.30:
                print("  ✅ Vibration fraction is in acceptable range (5-30%)")
            elif mean_fraction < 0.05:
                print("  ⚠️ Vibration fraction is low (<5%) - check filter")
            else:
                print("  ⚠️ Vibration fraction is high (>30%) - check for maneuvers")
        
        return metrics


class VibrationValidator:
    """
    Validate vibration extraction results against physical constraints.
    """
    
    def __init__(self, max_ratio=0.20):
        """
        Initialize validator.
        
        Parameters:
        -----------
        max_ratio : float
            Maximum allowed ratio of vibration RMS to global STD
        """
        self.max_ratio = max_ratio
    
    def validate(self, vib_rms, global_std, axis_names=None):
        """
        Validate vibration RMS values.
        
        Parameters:
        -----------
        vib_rms : np.ndarray
            Vibration RMS per window (n_windows, 3)
        global_std : np.ndarray
            Global standard deviation per axis (3,)
        axis_names : list
            Axis names for reporting
        
        Returns:
        --------
        dict
            Validation results
        """
        if axis_names is None:
            axis_names = ['X', 'Y', 'Z']
        
        results = {}
        all_passed = True
        
        print("\n" + "=" * 70)
        print("VIBRATION VALIDATION")
        print("=" * 70)
        
        for i, axis in enumerate(axis_names):
            mean_vib = np.mean(vib_rms[:, i])
            std_vib = np.std(vib_rms[:, i])
            ratio = mean_vib / global_std[i] if global_std[i] > 0 else np.inf
            
            # Check constraints
            passed_ratio = ratio < self.max_ratio
            passed_positive = mean_vib > 0.001
            passed_plausible = mean_vib < 1.0  # m/s² for accel
            
            axis_passed = passed_ratio and passed_positive and passed_plausible
            all_passed = all_passed and axis_passed
            
            results[axis] = {
                'mean_vibration': mean_vib,
                'std_vibration': std_vib,
                'global_std': global_std[i],
                'ratio': ratio,
                'passed_ratio': passed_ratio,
                'passed_positive': passed_positive,
                'passed_plausible': passed_plausible,
                'passed': axis_passed
            }
            
            status = "✅ PASS" if axis_passed else "❌ FAIL"
            print(f"\n  Axis {axis}: {status}")
            print(f"    Mean Vibration : {mean_vib:.4f}")
            print(f"    Global STD     : {global_std[i]:.4f}")
            print(f"    Ratio          : {ratio:.4f} (max: {self.max_ratio})")
            print(f"    Positive Check : {'✅' if passed_positive else '❌'}")
            print(f"    Plausible Check: {'✅' if passed_plausible else '❌'}")
        
        results['all_passed'] = all_passed
        
        print("\n" + "=" * 70)
        if all_passed:
            print("✅ VIBRATION VALIDATION PASSED")
        else:
            print("❌ VIBRATION VALIDATION FAILED")
            print("   → Adjust filter parameters and re-extract")
        print("=" * 70)
        
        return results


# ============================================================================
# TEST FUNCTION
# ============================================================================

def test_vibration_extractor():
    """
    Test the corrected vibration extractor on synthetic data.
    """
    print("\n" + "=" * 70)
    print("TESTING VIBRATION EXTRACTOR")
    print("=" * 70)
    
    # Generate synthetic data
    fs = 200.0
    duration = 60.0
    t = np.linspace(0, duration, int(duration * fs))
    
    # Low-frequency motion (0.5 Hz)
    motion = 2.0 * np.sin(2 * np.pi * 0.5 * t)
    
    # High-frequency vibration (20 Hz)
    vibration = 0.1 * np.sin(2 * np.pi * 20 * t)
    
    # Noise
    noise = 0.01 * np.random.randn(len(t))
    
    # Combined signal
    signal_combined = np.column_stack([
        motion + vibration + noise,
        motion + vibration + noise,
        motion + vibration + noise
    ])
    
    # Extract vibration
    extractor = VibrationExtractor(fs=fs, low_cut=5.0, high_cut=50.0, order=6)
    vib_rms, window_times = extractor.extract_windowed_rms(
        signal_combined, window_sec=10.0
    )
    
    # Verify
    print("\n📊 Verification:")
    extractor.verify_extraction(signal_combined)
    
    # Validate
    global_std = np.std(signal_combined, axis=0)
    validator = VibrationValidator(max_ratio=0.20)
    validator.validate(vib_rms, global_std)
    
    return extractor, vib_rms


if __name__ == "__main__":
    extractor, vib_rms = test_vibration_extractor()