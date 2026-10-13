#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path("/home/fatima/Research")

INPUT_FILE = (
    BASE_DIR
    / "validation"
    / "imu_degradation_results"
    / "window_feature_analysis"
    / "psd_analysis"
    / "stage2c"
    / "flight2_3_5_stage2c_window_spectral_features.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "validation"
    / "imu_degradation_results"
    / "window_feature_analysis"
    / "psd_analysis"
    / "stage3_vibration"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Load
# ============================================================

df = pd.read_csv(INPUT_FILE)

print("=" * 70)
print("STAGE 3 - VIBRATION CANDIDATE ANALYSIS")
print("=" * 70)

print(f"Input rows: {len(df)}")
print()


# ============================================================
# Build 10-50 Hz power features
# ============================================================

for sensor in ["accel", "gyro"]:

    subset = df[
        df["sensor"] == sensor
    ].copy()

    # Power 10-20 + 20-50
    subset["power_10_50hz"] = (
        subset["power_10_20hz"]
        + subset["power_20_50hz"]
    )

    subset["rms_10_50hz"] = np.sqrt(
        np.maximum(
            subset["power_10_50hz"],
            0
        )
    )

    # Keep 20-100 as comparison
    subset["vibration_20_100"] = (
        subset["hf_rms_20_100hz"]
    )

    # Keep 50-100 only as diagnostic
    subset["vibration_50_100"] = (
        subset["hf_rms_50_100hz"]
    )

    # Rename for easier aggregation
    subset = subset.rename(
        columns={
            "rms_10_50hz":
                "vibration_10_50",
        }
    )

    # --------------------------------------------------------
    # Average x/y/z
    # --------------------------------------------------------

    grouped = (
        subset
        .groupby(
            [
                "flight",
                "window_id",
                "window_mid_s",
            ]
        )
        .agg(
            vibration_10_50=(
                "vibration_10_50",
                "mean"
            ),

            vibration_20_100=(
                "vibration_20_100",
                "mean"
            ),

            vibration_50_100=(
                "vibration_50_100",
                "mean"
            ),

            spectral_centroid=(
                "spectral_centroid_hz",
                "mean"
            ),

            spectral_95=(
                "spectral_95_hz",
                "mean"
            ),

            total_rms=(
                "total_rms",
                "mean"
            ),
        )
        .reset_index()
    )

    grouped["sensor"] = sensor

    output = (
        OUTPUT_DIR
        / f"{sensor}_vibration_candidates.csv"
    )

    grouped.to_csv(
        output,
        index=False,
    )

    print(
        f"Saved: {output}"
    )

    # ========================================================
    # Summary statistics
    # ========================================================

    summary = (
        grouped
        .groupby("flight")
        [
            [
                "vibration_10_50",
                "vibration_20_100",
                "vibration_50_100",
                "spectral_centroid",
                "spectral_95",
                "total_rms",
            ]
        ]
        .agg(
            [
                "median",
                "mean",
                "std",
                "min",
                "max",
            ]
        )
    )

    summary_file = (
        OUTPUT_DIR
        / f"{sensor}_vibration_summary.csv"
    )

    summary.to_csv(summary_file)

    print(
        f"Saved: {summary_file}"
    )

    # ========================================================
    # Within-flight time correlations
    # ========================================================

    trend_records = []

    for flight in grouped["flight"].unique():

        flight_data = grouped[
            grouped["flight"] == flight
        ]

        for feature in [
            "vibration_10_50",
            "vibration_20_100",
            "vibration_50_100",
            "spectral_centroid",
        ]:

            valid = flight_data[
                [
                    "window_mid_s",
                    feature,
                ]
            ].dropna()

            if len(valid) >= 5:

                rho, p_value = spearmanr(
                    valid["window_mid_s"],
                    valid[feature],
                )

            else:

                rho = np.nan
                p_value = np.nan

            trend_records.append(
                {
                    "flight": flight,
                    "sensor": sensor,
                    "feature": feature,
                    "spearman_rho_time": rho,
                    "p_value": p_value,
                    "n_windows": len(valid),
                }
            )

    trends = pd.DataFrame(
        trend_records
    )

    trends_file = (
        OUTPUT_DIR
        / f"{sensor}_vibration_time_trends.csv"
    )

    trends.to_csv(
        trends_file,
        index=False,
    )

    print(
        f"Saved: {trends_file}"
    )

    # ========================================================
    # Plots
    # ========================================================

    for feature in [
        "vibration_10_50",
        "vibration_20_100",
        "vibration_50_100",
    ]:

        for flight in grouped["flight"].unique():

            flight_data = grouped[
                grouped["flight"] == flight
            ]

            plt.figure(
                figsize=(10, 5)
            )

            plt.plot(
                flight_data["window_mid_s"],
                flight_data[feature],
                marker="o",
                markersize=3,
                linewidth=1,
            )

            plt.xlabel(
                "Time [s]"
            )

            plt.ylabel(
                feature
            )

            plt.title(
                f"{flight} - {sensor} - {feature}"
            )

            plt.grid(
                True,
                alpha=0.3,
            )

            plt.tight_layout()

            filename = (
                f"{flight}_{sensor}_{feature}.png"
            )

            plt.savefig(
                OUTPUT_DIR / filename,
                dpi=200,
            )

            plt.close()


print()
print("=" * 70)
print("STAGE 3 COMPLETED")
print("=" * 70)