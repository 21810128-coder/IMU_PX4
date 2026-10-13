#!/usr/bin/env python3
"""
================================================================================
STEP 6: VIBRATION BAND SELECTION
================================================================================
This script analyzes the PSD results and selects the final vibration bands
based on the data-driven analysis.

Decision criteria:
1. Identify frequency bands with significant energy
2. Separate mechanical vibration from electronic noise
3. Select bands that are consistent across flights
4. Document the rationale for band selection

Author: Research Team
Date: 2024
================================================================================
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


# ============================================================
# CONFIGURATION
# ============================================================

PSD_DIR = Path(
    "/home/fatima/Research/validation/"
    "imu_degradation_results/window_feature_analysis/psd_analysis"
)

OUTPUT_DIR = Path(
    "/home/fatima/Research/validation/"
    "imu_degradation_results/vibration_features"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Candidate bands for exploration
CANDIDATE_BANDS = {
    "motion": (0.1, 1.0),
    "low_maneuver": (1.0, 5.0),
    "vibration_low": (5.0, 10.0),
    "vibration_mid": (10.0, 20.0),
    "vibration_high": (20.0, 50.0),
    "noise_low": (50.0, 75.0),
    "noise_high": (75.0, 100.0),
}

# Flights
FLIGHTS = ["F2", "F3", "F5"]


# ============================================================
# LOAD PSD RESULTS
# ============================================================

def load_psd_data():
    """Load PSD results for all flights."""
    print("\n" + "=" * 70)
    print("LOADING PSD DATA")
    print("=" * 70)
    
    psd_data = {}
    
    for flight in FLIGHTS:
        path = PSD_DIR / f"{flight.lower()}_median_psd.csv"
        
        if not path.exists():
            print(f"⚠️  File not found: {path}")
            continue
        
        df = pd.read_csv(path)
        psd_data[flight] = df
        
        print(f"✅ Loaded {flight}: {len(df)} frequency points")
        print(f"   Frequency range: {df['frequency_hz'].min():.2f} - "
              f"{df['frequency_hz'].max():.2f} Hz")
    
    return psd_data


# ============================================================
# ANALYZE BAND ENERGY
# ============================================================

def compute_band_energy(freqs, psd, bands):
    """Compute energy in each band."""
    energies = {}
    total_energy = np.trapezoid(psd, freqs)
    
    for band_name, (low, high) in bands.items():
        upper = min(high, freqs.max())
        mask = (freqs >= low) & (freqs < upper)
        
        if mask.sum() < 2:
            energies[band_name] = {
                'power': 0.0,
                'fraction': 0.0
            }
            continue
        
        power = np.trapezoid(psd[mask], freqs[mask])
        fraction = power / total_energy if total_energy > 0 else 0
        
        energies[band_name] = {
            'power': power,
            'fraction': fraction
        }
    
    return energies


def analyze_band_energy(psd_data):
    """Analyze band energy for all flights and axes."""
    print("\n" + "=" * 70)
    print("BAND ENERGY ANALYSIS")
    print("=" * 70)
    
    all_results = []
    
    for flight, df in psd_data.items():
        freqs = df['frequency_hz'].to_numpy()
        
        # Analyze each axis
        axis_names = [c for c in df.columns if c.endswith('_psd')]
        
        for axis_col in axis_names:
            psd = df[axis_col].to_numpy()
            
            # Skip if no data
            if np.all(np.isnan(psd)):
                continue
            
            # Remove NaN
            mask = np.isfinite(psd) & np.isfinite(freqs)
            freqs_clean = freqs[mask]
            psd_clean = psd[mask]
            
            # Compute band energies
            energies = compute_band_energy(freqs_clean, psd_clean, CANDIDATE_BANDS)
            
            # Build result row
            for band_name, data in energies.items():
                all_results.append({
                    'flight': flight,
                    'axis': axis_col.replace('_psd', ''),
                    'band': band_name,
                    'band_low': CANDIDATE_BANDS[band_name][0],
                    'band_high': CANDIDATE_BANDS[band_name][1],
                    'power': data['power'],
                    'fraction': data['fraction']
                })
    
    return pd.DataFrame(all_results)


def print_band_summary(band_df):
    """Print band energy summary."""
    print("\n" + "=" * 70)
    print("BAND ENERGY SUMMARY (Power Fraction)")
    print("=" * 70)
    
    # Pivot table
    pivot = band_df.pivot_table(
        values='fraction',
        index='band',
        columns='flight',
        aggfunc='mean'
    )
    
    # Reorder bands
    band_order = list(CANDIDATE_BANDS.keys())
    pivot = pivot.reindex(band_order)
    
    # Format as percentage
    pivot_pct = pivot * 100
    
    print("\n📊 Mean Power Fraction (%) by Flight:")
    print(pivot_pct.round(2).to_string())
    
    # Identify dominant bands
    print("\n📊 Dominant Band per Flight:")
    for flight in pivot.columns:
        dominant_band = pivot[flight].idxmax()
        dominant_value = pivot[flight].max() * 100
        print(f"   {flight}: {dominant_band} ({dominant_value:.1f}%)")
    
    return pivot


# ============================================================
# VISUALIZATION
# ============================================================

def plot_band_comparison(band_df, save_path):
    """Plot band energy comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    flights = ['F2', 'F3', 'F5']
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    bands = list(CANDIDATE_BANDS.keys())
    
    for idx, (flight, color) in enumerate(zip(flights, colors)):
        ax = axes[idx]
        
        # Filter data for this flight
        flight_data = band_df[band_df['flight'] == flight]
        
        if len(flight_data) == 0:
            ax.text(0.5, 0.5, f'No data for {flight}',
                   ha='center', va='center', transform=ax.transAxes)
            continue
        
        # Group by band
        band_means = flight_data.groupby('band')['fraction'].mean()
        band_means = band_means.reindex(bands)
        
        # Plot
        x = np.arange(len(bands))
        ax.bar(x, band_means.values * 100, color=color, alpha=0.8,
               edgecolor='black', linewidth=1)
        
        ax.set_xticks(x)
        ax.set_xticklabels(bands, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Power Fraction (%)', fontsize=11)
        ax.set_title(f'{flight} - Band Power Distribution', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Highlight vibration bands
        for i, band in enumerate(bands):
            if 'vibration' in band:
                ax.axvspan(i - 0.4, i + 0.4, alpha=0.15, color='red')
    
    plt.suptitle('Band Power Distribution Across Flights',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Figure saved: {save_path}")


# ============================================================
# BAND SELECTION DECISION
# ============================================================

def select_vibration_bands(band_df):
    """
    Select final vibration bands based on data.
    
    Returns:
    --------
    dict
        Selected bands with rationale
    """
    print("\n" + "=" * 70)
    print("VIBRATION BAND SELECTION")
    print("=" * 70)
    
    # Compute mean fraction per band per flight
    pivot = band_df.pivot_table(
        values='fraction',
        index='band',
        columns='flight',
        aggfunc='mean'
    )
    
    # Decision criteria
    selected_bands = {}
    
    print("\n📊 Decision Criteria:")
    print("-" * 50)
    
    # 1. Mechanical vibration band (5-50 Hz)
    vibration_bands = ['vibration_low', 'vibration_mid', 'vibration_high']
    
    # Check each flight
    for flight in pivot.columns:
        vibration_fraction = sum(
            pivot.loc[b, flight] for b in vibration_bands 
            if b in pivot.index
        )
        print(f"   {flight}: Mechanical vibration (5-50 Hz) = "
              f"{vibration_fraction*100:.1f}%")
    
    # 2. High-frequency band (50-100 Hz)
    noise_bands = ['noise_low', 'noise_high']
    
    for flight in pivot.columns:
        noise_fraction = sum(
            pivot.loc[b, flight] for b in noise_bands 
            if b in pivot.index
        )
        print(f"   {flight}: High-frequency (50-100 Hz) = "
              f"{noise_fraction*100:.1f}%")
    
    # 3. Low-frequency band (0.1-5 Hz)
    motion_bands = ['motion', 'low_maneuver']
    
    for flight in pivot.columns:
        motion_fraction = sum(
            pivot.loc[b, flight] for b in motion_bands 
            if b in pivot.index
        )
        print(f"   {flight}: Motion (0.1-5 Hz) = "
              f"{motion_fraction*100:.1f}%")
    
    # Final selection
    print("\n📊 Final Selection:")
    print("-" * 50)
    
    selected_bands = {
        'primary_vibration': {
            'name': 'Mechanical Vibration',
            'range': (5.0, 50.0),
            'rationale': 'Captures motor and propeller vibrations',
            'applies_to': ['F3', 'F5'],
            'unit_accel': '(m/s²)',
            'unit_gyro': '(rad/s)'
        },
        'secondary_vibration': {
            'name': 'High-Frequency Content',
            'range': (50.0, 100.0),
            'rationale': 'Captures electronic noise and structural resonance',
            'applies_to': ['F2'],
            'unit_accel': '(m/s²)',
            'unit_gyro': '(rad/s)'
        },
        'motion': {
            'name': 'Aircraft Motion',
            'range': (0.1, 5.0),
            'rationale': 'Captures maneuvers and slow dynamics',
            'applies_to': 'all',
            'unit_accel': '(m/s²)',
            'unit_gyro': '(rad/s)'
        }
    }
    
    for band_key, band_info in selected_bands.items():
        print(f"\n   {band_info['name']}:")
        print(f"      Range: {band_info['range'][0]} - {band_info['range'][1]} Hz")
        print(f"      Rationale: {band_info['rationale']}")
        print(f"      Applies to: {band_info['applies_to']}")
    
    return selected_bands


# ============================================================
# MAIN
# ============================================================

def main():
    """Main execution."""
    print("=" * 70)
    print("STEP 6: VIBRATION BAND SELECTION")
    print("=" * 70)
    
    # Load PSD data
    psd_data = load_psd_data()
    
    if not psd_data:
        print("❌ No PSD data found. Run PSD analysis first.")
        return
    
    # Analyze band energy
    band_df = analyze_band_energy(psd_data)
    
    # Save band energy
    band_path = OUTPUT_DIR / "band_energy_analysis.csv"
    band_df.to_csv(band_path, index=False)
    print(f"\n✅ Band energy saved: {band_path}")
    
    # Print summary
    pivot = print_band_summary(band_df)
    
    # Plot comparison
    fig_path = OUTPUT_DIR / "band_power_comparison.png"
    plot_band_comparison(band_df, fig_path)
    
    # Select bands
    selected_bands = select_vibration_bands(band_df)
    
    # Save selection
    import json
    selection_path = OUTPUT_DIR / "selected_vibration_bands.json"
    with open(selection_path, 'w') as f:
        json.dump(selected_bands, f, indent=2, default=str)
    print(f"\n✅ Band selection saved: {selection_path}")
    
    print("\n" + "=" * 70)
    print("✅ STEP 6 COMPLETED")
    print("=" * 70)
    
    return selected_bands


if __name__ == "__main__":
    selected_bands = main()