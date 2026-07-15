"""
Convert PlanetScope digital numbers to surface reflectance and calculate the spectral indices used as model features
, including NDVI, SAVI, GRVI, NDWI, BI, and NDBI. Since PlanetScope imagery lacks a SWIR band, the red band is used as a proxy for NDBI calculation.
"""

import numpy as np
import pandas as pd

# --- CONFIG ---
INPUT_CSV = "data/dataWithExtractedBands/extracted_pixel_values.csv"
OUTPUT_CSV = "data/dataWithExtractedBands/pixel_values_with_indices.csv"

PLANETSCOPE_SCALE_FACTOR = 10_000
SAVI_L = 0.5


def compute_indices(df):
    df["blue_refl"] = df["band_1"] / PLANETSCOPE_SCALE_FACTOR
    df["green_refl"] = df["band_2"] / PLANETSCOPE_SCALE_FACTOR
    df["red_refl"] = df["band_3"] / PLANETSCOPE_SCALE_FACTOR
    df["nir_refl"] = df["band_4"] / PLANETSCOPE_SCALE_FACTOR

    df["NDVI"] = (df["nir_refl"] - df["red_refl"]) / (df["nir_refl"] + df["red_refl"])
    df["SAVI"] = ((df["nir_refl"] - df["red_refl"]) / (df["nir_refl"] + df["red_refl"] + SAVI_L)) * (1 + SAVI_L)
    df["GRVI"] = (df["green_refl"] - df["red_refl"]) / (df["green_refl"] + df["red_refl"])
    df["NDWI"] = (df["green_refl"] - df["nir_refl"]) / (df["green_refl"] + df["nir_refl"])
    df["BI"] = np.sqrt((df["red_refl"] ** 2 + df["nir_refl"] ** 2) / 2)
    df["NDBI"] = (df["red_refl"] - df["nir_refl"]) / (df["red_refl"] + df["nir_refl"])  # modified

    return df


def main():
    df = pd.read_csv(INPUT_CSV)
    df = compute_indices(df)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(df)} rows with computed indices to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
