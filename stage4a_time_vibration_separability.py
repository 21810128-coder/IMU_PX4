
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import pearsonr, spearmanr

# ============================================================
# Stage 4A: Time-Vibration Separability Analysis
# Dependencies: numpy, pandas, scipy, matplotlib
# ============================================================

BASE = Path(
    "/home/fatima/Research/validation/"
    "imu_degradation_results/window_feature_analysis/"
    "psd_analysis/stage3_vibration"
)

OUT = BASE / "stage4a_separability"
OUT.mkdir(parents=True, exist_ok=True)

FILES = {
    "accel": BASE / "accel_vibration_candidates.csv",
    "gyro": BASE / "gyro_vibration_candidates.csv",
}

PRIMARY_V = "vibration_10_50"

V_CANDIDATES = [
    "vibration_10_50",
    "vibration_20_100",
    "vibration_50_100",
    "spectral_centroid",
]

TIME_CANDIDATES = [
    "time_s",
    "elapsed_time_s",
    "window_time_s",
    "window_start_s",
    "time",
    "t",
]

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------


def find_time_column(df):
    """Identify the timestamp representing each window midpoint."""

    time_candidates = [
        "window_mid_s",   # Actual column in the current CSV
        "time_s",
        "elapsed_time_s",
        "window_time_s",
        "window_start_s",
        "time",
        "t",
    ]

    for col in time_candidates:
        if col in df.columns:
            return col

    raise ValueError(
        "Could not identify the time column. "
        f"Available columns: {df.columns.tolist()}"
    )



def find_flight_column(df):
    """Find the flight identifier column."""

    for col in ["flight", "flight_id", "flight_name"]:
        if col in df.columns:
            return col

    raise ValueError(
        "Could not identify the flight column.\n"
        f"Available columns: {df.columns.tolist()}"
    )


def classify_correlation(rho):
    """Descriptive interpretation of correlation magnitude."""

    if not np.isfinite(rho):
        return "undefined"

    a = abs(rho)

    if a < 0.20:
        return "negligible"
    elif a < 0.40:
        return "weak"
    elif a < 0.60:
        return "moderate"
    elif a < 0.80:
        return "strong"
    else:
        return "very_strong"


def calculate_vif_condition(time_values, vibration_values):
    """
    Calculate VIF and condition number for two predictors:
        centered time
        vibration

    For two predictors, both VIF values equal:
        1 / (1 - r^2)

    Condition number is calculated from the standardized
    two-column predictor matrix.
    """

    X = np.column_stack([
        np.asarray(time_values, dtype=float),
        np.asarray(vibration_values, dtype=float),
    ])

    X = X[np.isfinite(X).all(axis=1)]

    if X.shape[0] < 4:
        return np.nan, np.nan, np.nan

    stds = X.std(axis=0, ddof=0)

    if np.any(stds < 1e-12):
        return np.nan, np.nan, np.nan

    Z = (X - X.mean(axis=0)) / stds

    r = float(np.corrcoef(Z, rowvar=False)[0, 1])
    r = float(np.clip(r, -1.0, 1.0))

    vif = 1.0 / max(1.0 - r**2, 1e-15)

    condition_number = float(np.linalg.cond(Z))

    return r, vif, condition_number


def safe_correlations(x, y):
    """Calculate Pearson and Spearman statistics safely."""

    if len(x) < 3:
        return np.nan, np.nan, np.nan, np.nan

    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan, np.nan, np.nan, np.nan

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        p_result = pearsonr(x, y)
        s_result = spearmanr(x, y)

    return (
        float(p_result.statistic),
        float(p_result.pvalue),
        float(s_result.statistic),
        float(s_result.pvalue),
    )


def assessment_from_vif(vif):
    if not np.isfinite(vif):
        return "undefined"

    if vif < 2:
        return "low"
    elif vif < 5:
        return "moderate"
    elif vif < 10:
        return "high"
    else:
        return "very_high"


# ------------------------------------------------------------
# Main analysis
# ------------------------------------------------------------

correlation_results = []
diagnostic_results = []
plot_data = []

for sensor, file_path in FILES.items():

    print("\n" + "=" * 70)
    print(f"Sensor: {sensor}")
    print(f"Input: {file_path}")
    print("=" * 70)

    if not file_path.exists():
        print(f"WARNING: Input file not found: {file_path}")
        continue

    df = pd.read_csv(file_path)

    if df.empty:
        print("WARNING: Input file is empty.")
        continue

    print(f"Rows: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")

    flight_col = find_flight_column(df)
    time_col = find_time_column(df)

    # Convert time to numeric
    df[time_col] = pd.to_numeric(
        df[time_col], errors="coerce"
    )

    for flight in sorted(df[flight_col].dropna().unique()):

        flight_df = df[df[flight_col] == flight].copy()

        # Sort by time to make plots and inspection consistent
        flight_df = flight_df.sort_values(time_col)

        print(
            f"\n{sensor} | {flight} | "
            f"{len(flight_df)} rows"
        )

        for vcol in V_CANDIDATES:

            if vcol not in flight_df.columns:
                print(f"  Skipping missing feature: {vcol}")
                continue

            data = flight_df[[time_col, vcol]].copy()

            data[time_col] = pd.to_numeric(
                data[time_col], errors="coerce"
            )
            data[vcol] = pd.to_numeric(
                data[vcol], errors="coerce"
            )

            data = data.replace(
                [np.inf, -np.inf], np.nan
            ).dropna()

            if len(data) < 4:
                print(
                    f"  Too few valid rows for {vcol}: "
                    f"{len(data)}"
                )
                continue

            # Center time within this flight.
            # Centering changes the intercept, not the correlation.
            t = data[time_col].to_numpy(dtype=float)
            V = data[vcol].to_numpy(dtype=float)
            t_centered = t - np.mean(t)

            (
                pearson_r,
                pearson_p,
                spearman_rho,
                spearman_p,
            ) = safe_correlations(t_centered, V)

            _, vif, condition_number = (
                calculate_vif_condition(t_centered, V)
            )

            corr_result = {
                "sensor": sensor,
                "flight": str(flight),
                "time_column": time_col,
                "vibration_feature": vcol,
                "n_windows": len(data),
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
                "spearman_rho": spearman_rho,
                "spearman_p": spearman_p,
                "correlation_strength": classify_correlation(
                    spearman_rho
                ),
                "VIF_time": vif,
                "VIF_vibration": vif,
                "condition_number": condition_number,
                "VIF_assessment": assessment_from_vif(vif),
            }

            correlation_results.append(corr_result)

            diagnostic_results.append({
                "sensor": sensor,
                "flight": str(flight),
                "vibration_feature": vcol,
                "n_windows": len(data),
                "pearson_r": pearson_r,
                "spearman_rho": spearman_rho,
                "VIF": vif,
                "condition_number": condition_number,
                "VIF_assessment": assessment_from_vif(vif),
            })

            # Save plotting data for primary candidate
            if vcol == PRIMARY_V:
                plot_data.append({
                    "sensor": sensor,
                    "flight": str(flight),
                    "time": t,
                    "vibration": V,
                })

            print(
                f"  {vcol}: "
                f"rho={spearman_rho:.4f}, "
                f"VIF={vif:.3f}, "
                f"condition={condition_number:.3f}"
            )


# ------------------------------------------------------------
# Save CSV results
# ------------------------------------------------------------

corr_df = pd.DataFrame(correlation_results)
diag_df = pd.DataFrame(diagnostic_results)

corr_file = OUT / "stage4a_time_vibration_correlations.csv"
diag_file = OUT / "stage4a_vif_condition_number.csv"
primary_file = OUT / "stage4a_primary_candidate.csv"

if not corr_df.empty:
    corr_df.to_csv(corr_file, index=False)

    primary_df = corr_df[
        corr_df["vibration_feature"] == PRIMARY_V
    ].copy()

    primary_df.to_csv(primary_file, index=False)

if not diag_df.empty:
    diag_df.to_csv(diag_file, index=False)

print("\n" + "=" * 70)
print("OUTPUT FILES")
print("=" * 70)

for path in [corr_file, diag_file, primary_file]:
    print(f"{path}: {'created' if path.exists() else 'not created'}")


# ------------------------------------------------------------
# Plot time versus primary vibration feature
# ------------------------------------------------------------

for item in plot_data:

    sensor = item["sensor"]
    flight = item["flight"]
    t = item["time"]
    V = item["vibration"]

    plt.figure(figsize=(8, 5))

    plt.scatter(t, V, s=25, alpha=0.75)

    plt.xlabel("Time within flight (s)")
    plt.ylabel("Vibration RMS (10-50 Hz)")
    plt.title(
        f"{sensor.upper()} | {flight}: Time vs Vibration"
    )

    plt.grid(True, alpha=0.25)
    plt.tight_layout()

    plot_file = OUT / (
        f"{sensor}_{flight}_time_vs_vibration_10_50.png"
    )

    plt.savefig(plot_file, dpi=200)
    plt.close()


# ------------------------------------------------------------
# Final console summary
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STAGE 4A SUMMARY — PRIMARY FEATURE")
print("=" * 70)

if not corr_df.empty:

    summary = corr_df[
        corr_df["vibration_feature"] == PRIMARY_V
    ].copy()

    if not summary.empty:
        display_cols = [
            "sensor",
            "flight",
            "n_windows",
            "pearson_r",
            "spearman_rho",
            "spearman_p",
            "VIF_time",
            "condition_number",
            "correlation_strength",
            "VIF_assessment",
        ]

        print(
            summary[display_cols]
            .sort_values(["sensor", "flight"])
            .to_string(index=False)
        )
    else:
        print(
            f"No rows found for primary feature: {PRIMARY_V}"
        )

else:
    print(
        "No results generated. Check input paths and column names."
    )

print("\nInterpretation guidance:")
print("  VIF near 1 : low linear collinearity")
print("  VIF 1-5    : generally limited/moderate")
print("  VIF 5-10   : high; investigate")
print("  VIF >= 10  : very high; coefficient estimates may be unstable")
print()
print(
    "Important: correlation and VIF diagnose predictor overlap; "
    "they do not prove physical sensor degradation."
)
print("Stage 4A finished.")
