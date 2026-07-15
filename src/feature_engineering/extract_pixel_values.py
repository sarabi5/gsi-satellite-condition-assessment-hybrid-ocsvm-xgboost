"""
Extract valid pixel-level PlanetScope band values from clipped GeoTIFFs and link them with GSI site metadata. 
The four spectral bands (blue, green, red, and NIR) are exported to CSV for downstream reflectance conversion and index calculation.
"""

import os

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask
from shapely.geometry import box

# --- CONFIG ---
POLYGON_SHAPEFILE = "data/filtered_gsi_polygons.shp"
SITE_MATCH_CSV = "data/gsi_planet_final_results.csv"  # SMPId, image_id columns
CLIPPED_IMAGE_DIR = "data/dataWithExtractedBands/clipped_images"
OUTPUT_CSV = "data/dataWithExtractedBands/extracted_pixel_values.csv"


def extract_pixel_values(tif_path, geometry, geometry_crs):
    """Return a list of per-pixel band-value dicts for valid pixels within `geometry`."""
    with rasterio.open(tif_path) as src:
        geom_gdf = gpd.GeoDataFrame([1], geometry=[geometry], crs=geometry_crs)
        if geom_gdf.crs != src.crs:
            geom_gdf = geom_gdf.to_crs(src.crs)
            geometry = geom_gdf.geometry.iloc[0]

        if not geometry.intersects(box(*src.bounds)):
            return []

        out_image, _ = mask(src, [geometry], crop=True, all_touched=True)
        n_bands = src.count

        valid_mask = np.ones(out_image[0].shape, dtype=bool)
        for band_idx in range(n_bands):
            band = out_image[band_idx]
            if src.nodata is not None:
                valid_mask &= band != src.nodata
            valid_mask &= ~np.isnan(band)
            valid_mask &= band != 0  # exclude likely nodata encoded as zero

        rows, cols = np.where(valid_mask)
        if len(rows) == 0:
            return []

        band_names = [f"band_{i + 1}" for i in range(n_bands)]
        return [
            {band_names[b]: float(out_image[b, r, c]) for b in range(n_bands)}
            for r, c in zip(rows, cols)
        ]


def main():
    gdf = gpd.read_file(POLYGON_SHAPEFILE)
    df = pd.read_csv(SITE_MATCH_CSV)

    all_records = []
    for _, row in df.iterrows():
        smp_id, image_id = row["SMPId"], row["image_id"]
        tif_path = os.path.join(CLIPPED_IMAGE_DIR, f"SMP_{smp_id}_{image_id}.tif")
        if not os.path.exists(tif_path):
            continue

        geom_rows = gdf[gdf["SMPId"] == smp_id]
        if geom_rows.empty:
            continue

        pixels = extract_pixel_values(tif_path, geom_rows.geometry.iloc[0], gdf.crs)
        base_row = row.to_dict()
        for px in pixels:
            record = base_row.copy()
            record.update(px)
            all_records.append(record)

    if not all_records:
        print("No pixel records extracted; check CLIPPED_IMAGE_DIR and input paths.")
        return

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    pd.DataFrame(all_records).to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(all_records)} pixel records to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
