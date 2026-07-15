"""
Aggregate pixel-level spectral data into site-level features and merge them with inspection ratings, antecedent rainfall, and site geometry to create the final model-ready dataset.

Step 1: Feature aggregation
For each spectral band and index, global quantile thresholds (25th, 50th, and 75th percentiles) are computed. The percentage of pixels within each quantile is then calculated for every site to generate site-level features.

Step 2: Data integration
Merge the site-level features with inspection ratings, 3-day antecedent rainfall, and site geometry (area and perimeter).
"""

from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# --- CONFIG ---
PIXEL_CSV = "data/dataWithExtractedBands/pixel_values_with_indices.csv"
QUANTILE_FEATURES_CSV = "data/quantile_percentages.csv"

INSPECTION_CSV = "data/AllInspections_Cleaned1.csv"
RAINFALL_DIR = "data/rainfall"  # ASOS station CSVs, e.g. PHL*.csv
GEOMETRY_XLS = "data/gsi_planet_final_results_excel.xls"

FINAL_OUTPUT_CSV = "data/quantile_ml_ready_dataset.csv"

FEATURE_COLUMNS = [
    "blue_refl", "green_refl", "red_refl", "nir_refl",
    "NDVI", "SAVI", "GRVI", "NDWI", "BI", "NDBI",
]

# Inspection target ranking columns retained for modeling
RANKING_COLUMNS = [
    "OverallSMPFunctionalRatingId",
    "VegetationRatingId(Ranking)",
    "SMPVegetativelyStabilizedId(Ranking)",
    "SignOfErosionId(Ranking)",
    "DrainageAreaRatingId(Ranking)",
    "DebrisTrashWithinDrainageAreaId(Ranking)",
]

# Step 1: Quantile-bin feature aggregation
def build_quantile_features(df):
    global_quantiles = {feature: df[feature].quantile([0.25, 0.5, 0.75]) for feature in FEATURE_COLUMNS}

    rows = []
    for smp_id, group in df.groupby("SMPId"):
        row_data = {"SMPId": smp_id}
        for feature in FEATURE_COLUMNS:
            quantiles = global_quantiles[feature]
            bins = [-np.inf] + quantiles.tolist() + [np.inf]

            for i in range(len(bins) - 1):
                lower, upper = bins[i], bins[i + 1]
                count = ((group[feature] > lower) & (group[feature] <= upper)).sum()
                pct = count / len(group)

                lower_str = f"{lower:.2f}" if lower != -np.inf else "-inf"
                upper_str = f"{upper:.2f}" if upper != np.inf else "inf"
                row_data[f"{lower_str}<{feature}<{upper_str}"] = pct

        rows.append(row_data)

    return pd.DataFrame(rows)

# Step 2: Merge with inspection, rainfall, and geometry data
def load_daily_rainfall(rainfall_dir):
    """Load and aggregate ASOS precipitation CSVs to daily totals.

    Source: Iowa Environmental Mesonet ASOS request tool
    (https://mesonet.agron.iastate.edu/request/download.phtml?network=PA_ASOS)
    'T' (trace) and 'M' (missing) values are dropped before summation.
    """
    files = [f for f in Path(rainfall_dir).glob("*.csv") if "PHL" in f.name]
    rainfall = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    rainfall = rainfall[~rainfall["p01m"].isin(["T", "M"])].drop_duplicates()
    rainfall["p01m"] = pd.to_numeric(rainfall["p01m"], errors="coerce")
    rainfall["date"] = pd.to_datetime(rainfall["valid"], errors="coerce").dt.date

    daily = rainfall.groupby("date")["p01m"].sum().reset_index()
    daily.columns = ["date", "daily_rainfall"]
    daily["date"] = pd.to_datetime(daily["date"])
    return daily


def calculate_3day_rainfall(inspection_date, daily_rainfall):
    """Mean daily rainfall over the 3 days preceding (not including) the inspection date."""
    if pd.isna(inspection_date):
        return np.nan
    end_date = inspection_date - timedelta(days=1)
    start_date = end_date - timedelta(days=2)
    window = daily_rainfall[(daily_rainfall["date"] >= start_date) & (daily_rainfall["date"] <= end_date)]
    return window["daily_rainfall"].mean() if not window.empty else 0.0


def prepare_inspection_data(inspection_csv):
    df = pd.read_csv(inspection_csv)
    df.columns = df.columns.str.strip()
    df["InspectionDate"] = pd.to_datetime(df["InspectionDate"], errors="coerce")

    keep_columns = ["SMPId", "InspectionDate", "DATE_INSTALLED", "GSI Type", "WeatherConditionDesc"] + RANKING_COLUMNS
    df = df[[c for c in keep_columns if c in df.columns]].copy()

    # Cast rating columns to a consistent integer/nullable type
    for col in RANKING_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.strip().replace("", np.nan), errors="coerce"
            ).round().astype("Int64")

    return df


def merge_all(quantile_features, inspection_csv, rainfall_dir, geometry_xls):
    inspection = prepare_inspection_data(inspection_csv)
    daily_rainfall = load_daily_rainfall(rainfall_dir)
    inspection["rainfall_3day"] = inspection["InspectionDate"].apply(
        lambda d: calculate_3day_rainfall(d, daily_rainfall)
    )

    merged = quantile_features.merge(inspection, on="SMPId", how="left")

    geometry = pd.read_excel(geometry_xls)
    geometry.columns = geometry.columns.str.strip()
    geom_cols = [c for c in ["SMPId", "Shape__Are", "Shape__Len"] if c in geometry.columns]
    geometry_subset = geometry[geom_cols].drop_duplicates(subset="SMPId")
    merged = merged.merge(geometry_subset, on="SMPId", how="left")

    return merged


def main():
    pixel_df = pd.read_csv(PIXEL_CSV)

    quantile_features = build_quantile_features(pixel_df)
    quantile_features.to_csv(QUANTILE_FEATURES_CSV, index=False)
    print(f"Saved site-level quantile features to {QUANTILE_FEATURES_CSV}")

    final_df = merge_all(quantile_features, INSPECTION_CSV, RAINFALL_DIR, GEOMETRY_XLS)
    final_df.to_csv(FINAL_OUTPUT_CSV, index=False)
    print(f"Saved final model-ready dataset ({len(final_df)} rows) to {FINAL_OUTPUT_CSV}")


if __name__ == "__main__":
    main()
