#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import welch
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path("/home/fatima/Research")

OUTPUT_DIR = (
    BASE_DIR
    / "validation"
    / "imu_degradation_results"
    / "window_feature_analysis"
    / "psd_analysis"
    / "stage2c"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FLIGHTS = {
    "F2": BASE_DIR / "validation/flight2/PX4/CSV/flight2_sensor_combined_0.csv",
    "F3": BASE_DIR / "validation/flight3/PX4/CSV/flight3_sensor_combined_0.csv",
    "F5": BASE_DIR / "validation/flight5/PX4/CSV/flight5_sensor_combined_0.csv",
}

WINDOW_SECONDS = 30.0

N_PER_SEG = 4096
NO_OVERLAP = 2048

# Frequency bands
BANDS = {
    "0_1": (0.0, 1.0),
    "1_5": (1.0, 5.0),
    "5_10": (5.0, 10.0),
    "10_20": (10.0, 20.0),
    "20_50": (20.0, 50.0),
    "50_100": (50.0, 100.0),
}

HF_BANDS = {
    "10_100": (10.0, 100.0),
    "20_100": (20.0, 100.0),
    "50_100": (50.0, 100.0),
}

AXES = {
    "accel": [
        "accelerometer_m_s2[0]",
        "accelerometer_m_s2[1]",
        "accelerometer_m_s2[2]",
    ],
    "gyro": [
        "gyro_rad[0]",
        "gyro_rad[1]",
        "gyro_rad[2]",
    ],
}


# ============================================================
# Helper functions
# ============================================================

def estimate_sampling_quality(timestamp_us):
    """
    Estimate sampling frequency and timestamp regularity.
    No interpolation or resampling is performed.
    """

    t = np.asarray(timestamp_us, dtype=float) / 1e6

    dt = np.diff(t)
    dt = dt[np.isfinite(dt)]
    dt = dt[dt > 0]

    if len(dt) == 0:
        return {
            "fs_hz": np.nan,
            "dt_median_s": np.nan,
            "dt_mean_s": np.nan,
            "jitter_std_pct": np.nan,
            "jitter_mad_pct": np.nan,
            "gaps_gt_2x": np.nan,
            "gaps_gt_5x": np.nan,
            "largest_gap_ms": np.nan,
        }

    dt_median = np.median(dt)
    dt_mean = np.mean(dt)

    fs = 1.0 / dt_median

    jitter_pct = ((dt - dt_median) / dt_median) * 100.0

    jitter_std_pct = np.std(jitter_pct)

    jitter_mad_pct = np.median(
        np.abs(jitter_pct - np.median(jitter_pct))
    )

    gaps_gt_2x = np.sum(dt > 2.0 * dt_median)
    gaps_gt_5x = np.sum(dt > 5.0 * dt_median)

    largest_gap_ms = np.max(dt) * 1000.0

    return {
        "fs_hz": fs,
        "dt_median_s": dt_median,
        "dt_mean_s": dt_mean,
        "jitter_std_pct": jitter_std_pct,
        "jitter_mad_pct": jitter_mad_pct,
        "gaps_gt_2x": int(gaps_gt_2x),
        "gaps_gt_5x": int(gaps_gt_5x),
        "largest_gap_ms": largest_gap_ms,
    }


def integrate_band(f, psd, low, high):
    """
    Integrate PSD over a frequency band.
    Returns variance/power in signal-units^2.
    """

    mask = (f >= low) & (f <= high)

    if np.sum(mask) < 2:
        return np.nan

    return np.trapezoid(psd[mask], f[mask])


def rms_from_power(power):
    if not np.isfinite(power) or power < 0:
        return np.nan

    return np.sqrt(power)


def spectral_features(signal, fs):
    """
    Compute Welch PSD and spectral features for one axis.
    """

    signal = np.asarray(signal, dtype=float)

    signal = signal[np.isfinite(signal)]

    if len(signal) < N_PER_SEG:
        return None

    nperseg = min(N_PER_SEG, len(signal))
    noverlap = min(NO_OVERLAP, nperseg // 2)

    f, psd = welch(
        signal,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="constant",
        scaling="density",
        average="median",
    )

    # --------------------------------------------------------
    # Total power
    # --------------------------------------------------------

    total_power = integrate_band(
        f,
        psd,
        0.0,
        min(100.0, fs / 2.0),
    )

    total_rms = rms_from_power(total_power)

    # --------------------------------------------------------
    # Band features
    # --------------------------------------------------------

    result = {
        "total_power": total_power,
        "total_rms": total_rms,
    }

    for band_name, (low, high) in BANDS.items():

        high_effective = min(high, fs / 2.0)

        if high_effective <= low:
            power = np.nan
        else:
            power = integrate_band(
                f,
                psd,
                low,
                high_effective,
            )

        result[f"power_{band_name}hz"] = power
        result[f"rms_{band_name}hz"] = rms_from_power(power)

        if np.isfinite(power) and np.isfinite(total_power) and total_power > 0:
            result[f"fraction_{band_name}hz"] = power / total_power
        else:
            result[f"fraction_{band_name}hz"] = np.nan

    # --------------------------------------------------------
    # High-frequency features
    # --------------------------------------------------------

    for name, (low, high) in HF_BANDS.items():

        high_effective = min(high, fs / 2.0)

        if high_effective <= low:
            power = np.nan
        else:
            power = integrate_band(
                f,
                psd,
                low,
                high_effective,
            )

        result[f"hf_power_{name}hz"] = power
        result[f"hf_rms_{name}hz"] = rms_from_power(power)

        if np.isfinite(power) and np.isfinite(total_power) and total_power > 0:
            result[f"hf_fraction_{name}hz"] = power / total_power
        else:
            result[f"hf_fraction_{name}hz"] = np.nan

    # --------------------------------------------------------
    # Spectral centroid
    # --------------------------------------------------------

    valid = (
        np.isfinite(f)
        & np.isfinite(psd)
        & (psd >= 0)
    )

    f_valid = f[valid]
    psd_valid = psd[valid]

    if len(f_valid) > 1:

        denominator = np.trapezoid(
            psd_valid,
            f_valid,
        )

        numerator = np.trapezoid(
            f_valid * psd_valid,
            f_valid,
        )

        if denominator > 0:
            centroid = numerator / denominator
        else:
            centroid = np.nan

        result["spectral_centroid_hz"] = centroid

        # ----------------------------------------------------
        # Spectral bandwidth
        # ----------------------------------------------------

        if np.isfinite(centroid):

            bandwidth_power = np.trapezoid(
                ((f_valid - centroid) ** 2) * psd_valid,
                f_valid,
            )

            if denominator > 0:
                result["spectral_bandwidth_hz"] = np.sqrt(
                    bandwidth_power / denominator
                )
            else:
                result["spectral_bandwidth_hz"] = np.nan

        else:
            result["spectral_bandwidth_hz"] = np.nan

    else:
        result["spectral_centroid_hz"] = np.nan
        result["spectral_bandwidth_hz"] = np.nan

    # --------------------------------------------------------
    # Spectral 95%
    # --------------------------------------------------------

    if len(f_valid) > 1:

        cumulative_power = np.concatenate(
            [
                [0.0],
                np.cumsum(
                    0.5
                    * (
                        psd_valid[1:]
                        + psd_valid[:-1]
                    )
                    * np.diff(f_valid)
                ),
            ]
        )

        if cumulative_power[-1] > 0:

            normalized = (
                cumulative_power
                / cumulative_power[-1]
            )

            idx = np.searchsorted(
                normalized,
                0.95,
            )

            if idx >= len(f_valid):
                idx = len(f_valid) - 1

            result["spectral_95_hz"] = f_valid[idx]

        else:
            result["spectral_95_hz"] = np.nan

    else:
        result["spectral_95_hz"] = np.nan

    # --------------------------------------------------------
    # Dominant frequency
    # --------------------------------------------------------

    # Ignore very-low-frequency content when searching
    # for a vibration peak.
    peak_mask = (
        (f_valid >= 0.5)
        & (f_valid <= min(100.0, fs / 2.0))
    )

    if np.any(peak_mask):

        f_peak = f_valid[peak_mask]
        psd_peak = psd_valid[peak_mask]

        peak_idx = np.argmax(psd_peak)

        result["dominant_frequency_hz"] = f_peak[peak_idx]

        result["dominant_psd"] = psd_peak[peak_idx]

    else:
        result["dominant_frequency_hz"] = np.nan
        result["dominant_psd"] = np.nan

    return result


def get_complete_windows(df):
    """
    Return complete 30-second windows using timestamps.
    """

    t = df["timestamp"].to_numpy(dtype=np.int64) / 1e6

    t0 = t[0]
    t_end = t[-1]

    n_windows = int(
        np.floor((t_end - t0) / WINDOW_SECONDS)
    )

    windows = []

    for i in range(n_windows):

        start = t0 + i * WINDOW_SECONDS
        end = start + WINDOW_SECONDS

        mask = (
            (t >= start)
            & (t < end)
        )

        window = df.loc[mask].copy()

        if len(window) < 2:
            continue

        actual_duration = (
            window["timestamp"].iloc[-1]
            - window["timestamp"].iloc[0]
        ) / 1e6

        # Require approximately the full 30 seconds.
        if actual_duration < WINDOW_SECONDS * 0.98:
            continue

        windows.append(
            (
                i + 1,
                start,
                end,
                window,
            )
        )

    return windows


# ============================================================
# Process one flight
# ============================================================

def process_flight(flight_name, csv_path):

    print()
    print("=" * 70)
    print(f"Processing {flight_name}")
    print(csv_path)
    print("=" * 70)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"File not found: {csv_path}"
        )

    df = pd.read_csv(csv_path)

    required_columns = (
        ["timestamp"]
        + AXES["accel"]
        + AXES["gyro"]
    )

    missing = [
        c for c in required_columns
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{flight_name}: missing columns: {missing}"
        )

    df = df[required_columns].copy()

    df = df.dropna(
        subset=required_columns
    )

    df = df.sort_values("timestamp")

    windows = get_complete_windows(df)

    print(f"Complete windows: {len(windows)}")

    records = []

    for (
        window_id,
        start_s,
        end_s,
        window,
    ) in windows:

        quality = estimate_sampling_quality(
            window["timestamp"].to_numpy()
        )

        fs = quality["fs_hz"]

        # We need enough samples for Welch.
        if len(window) < N_PER_SEG:
            print(
                f"Skipping {flight_name} "
                f"window {window_id}: "
                f"not enough samples"
            )
            continue

        for sensor_type, columns in AXES.items():

            for axis_index, column in enumerate(columns):

                signal = window[column].to_numpy(
                    dtype=float
                )

                features = spectral_features(
                    signal,
                    fs,
                )

                if features is None:
                    continue

                record = {
                    "flight": flight_name,
                    "window_id": window_id,
                    "window_start_s": start_s,
                    "window_end_s": end_s,
                    "window_mid_s": (
                        start_s + end_s
                    ) / 2.0,

                    "sensor": sensor_type,
                    "axis": ["x", "y", "z"][axis_index],

                    "n_samples": len(signal),

                    "fs_hz": quality["fs_hz"],
                    "dt_median_s": quality["dt_median_s"],
                    "dt_mean_s": quality["dt_mean_s"],

                    "jitter_std_pct": (
                        quality["jitter_std_pct"]
                    ),

                    "jitter_mad_pct": (
                        quality["jitter_mad_pct"]
                    ),

                    "gaps_gt_2x": (
                        quality["gaps_gt_2x"]
                    ),

                    "gaps_gt_5x": (
                        quality["gaps_gt_5x"]
                    ),

                    "largest_gap_ms": (
                        quality["largest_gap_ms"]
                    ),
                }

                record.update(features)

                records.append(record)

    return pd.DataFrame(records)


# ============================================================
# Main
# ============================================================

def main():

    all_results = []

    for flight_name, csv_path in FLIGHTS.items():

        result = process_flight(
            flight_name,
            csv_path,
        )

        all_results.append(result)

    results = pd.concat(
        all_results,
        ignore_index=True,
    )

    results = results.sort_values(
        [
            "flight",
            "window_id",
            "sensor",
            "axis",
        ]
    )

    # --------------------------------------------------------
    # Save full per-window dataset
    # --------------------------------------------------------

    output_csv = (
        OUTPUT_DIR
        / "flight2_3_5_stage2c_window_spectral_features.csv"
    )

    results.to_csv(
        output_csv,
        index=False,
    )

    print()
    print("Saved:")
    print(output_csv)

    # --------------------------------------------------------
    # Flight/window summary
    # --------------------------------------------------------

    summary_columns = [
        "flight",
        "window_id",
        "window_mid_s",
        "sensor",

        "hf_rms_10_100hz",
        "hf_fraction_10_100hz",

        "hf_rms_20_100hz",
        "hf_fraction_20_100hz",

        "hf_rms_50_100hz",
        "hf_fraction_50_100hz",

        "spectral_centroid_hz",
        "spectral_bandwidth_hz",
        "spectral_95_hz",
        "dominant_frequency_hz",

        "total_rms",
    ]

    summary = results[summary_columns].copy()

    summary_output = (
        OUTPUT_DIR
        / "flight2_3_5_stage2c_compact.csv"
    )

    summary.to_csv(
        summary_output,
        index=False,
    )

    print(summary_output)

    # --------------------------------------------------------
    # Median summary by flight and sensor
    # --------------------------------------------------------

    numeric_columns = [
        "hf_rms_10_100hz",
        "hf_fraction_10_100hz",

        "hf_rms_20_100hz",
        "hf_fraction_20_100hz",

        "hf_rms_50_100hz",
        "hf_fraction_50_100hz",

        "spectral_centroid_hz",
        "spectral_bandwidth_hz",
        "spectral_95_hz",
        "dominant_frequency_hz",

        "total_rms",
    ]

    flight_summary = (
        results
        .groupby(
            ["flight", "sensor"]
        )[numeric_columns]
        .median()
        .reset_index()
    )

    flight_summary_output = (
        OUTPUT_DIR
        / "flight2_3_5_stage2c_flight_summary.csv"
    )

    flight_summary.to_csv(
        flight_summary_output,
        index=False,
    )

    print(flight_summary_output)

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("STAGE 2C SUMMARY")
    print("=" * 70)

    print(
        flight_summary.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Time-series plots
    # --------------------------------------------------------

    plot_metrics = [
        (
            "hf_fraction_20_100hz",
            "HF fraction 20-100 Hz",
        ),
        (
            "spectral_centroid_hz",
            "Spectral centroid [Hz]",
        ),
        (
            "spectral_95_hz",
            "Spectral 95% [Hz]",
        ),
        (
            "hf_rms_20_100hz",
            "HF RMS 20-100 Hz",
        ),
    ]

    for flight in results["flight"].unique():

        for sensor in ["accel", "gyro"]:

            for metric, ylabel in plot_metrics:

                fig, ax = plt.subplots(
                    figsize=(10, 5)
                )

                subset = results[
                    (results["flight"] == flight)
                    & (results["sensor"] == sensor)
                ]

                for axis in ["x", "y", "z"]:

                    axis_data = subset[
                        subset["axis"] == axis
                    ]

                    ax.plot(
                        axis_data["window_mid_s"],
                        axis_data[metric],
                        marker="o",
                        markersize=3,
                        linewidth=1,
                        label=axis,
                    )

                ax.set_xlabel(
                    "Time [s]"
                )

                ax.set_ylabel(
                    ylabel
                )

                ax.set_title(
                    f"{flight} - "
                    f"{sensor} - "
                    f"{ylabel}"
                )

                ax.grid(
                    True,
                    alpha=0.3,
                )

                ax.legend()

                fig.tight_layout()

                filename = (
                    f"{flight}_{sensor}_"
                    f"{metric}.png"
                )

                fig.savefig(
                    OUTPUT_DIR / filename,
                    dpi=200,
                )

                plt.close(fig)

    print()
    print("Plots saved to:")
    print(OUTPUT_DIR)

    print()
    print("Stage 2C completed successfully.")


if __name__ == "__main__":
    main()