"""
Spatial distribution map of GSI sites.
"""

import math
import os
import tempfile
import zipfile

import contextily as ctx
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import requests
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

# --- CONFIG ---
POLYGON_SHAPEFILE = "data/filtered_gsi_polygons.shp"
SITE_DATA_CSV = "data/quantile_ml_ready_dataset.csv"
OUTPUT_PATH = "outputs/figures/GSI_map_philadelphia.png"

CENSUS_PLACES_URL = "https://www2.census.gov/geo/tiger/TIGER2022/PLACE/tl_2022_42_place.zip"
CITY_NAME = "Philadelphia"

BASIN_COLOR, BASIN_EDGE = "#C0392B", "#7B241C"
BIO_COLOR, BIO_EDGE = "#AED6F1", "#2E86C1"


def mercator_to_lon(x):
    return x * 180 / 20037508.34


def mercator_to_lat(y):
    return math.degrees(math.atan(math.sinh(y * math.pi / 20037508.34)))


def load_city_boundary(city_name, places_url):
    """Download and extract the US Census TIGER/Line place boundary for `city_name`."""
    response = requests.get(places_url)
    response.raise_for_status()

    tmp_dir = tempfile.mkdtemp()
    zip_path = os.path.join(tmp_dir, "places.zip")
    with open(zip_path, "wb") as f:
        f.write(response.content)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(tmp_dir)

    shp_path = next(f for f in os.listdir(tmp_dir) if f.endswith(".shp"))
    places = gpd.read_file(os.path.join(tmp_dir, shp_path))
    return places[places["NAME"] == city_name].to_crs(epsg=3857)


def load_site_geometries(polygon_shapefile, site_data_csv):
    """Load GSI polygons and restrict to sites present in the model dataset, tagged by GSI type."""
    gdf = gpd.read_file(polygon_shapefile)
    df = pd.read_csv(site_data_csv)

    gdf["SMPId"] = gdf["SMPId"].astype(float).astype(int).astype(str)
    df["SMPId"] = df["SMPId"].astype(float).astype(int).astype(str)

    matched = gdf[gdf["SMPId"].isin(set(df["SMPId"]))].copy()
    matched = matched.merge(df[["SMPId", "GSI Type"]].drop_duplicates(), on="SMPId", how="left")
    return matched.to_crs(epsg=3857)


def plot_site_distribution(gdf_matched, philly_boundary, output_path):
    fig, ax = plt.subplots(figsize=(16, 14), facecolor="white")

    # Plot the city boundary first to establish and lock the axis extent,
    # then add the basemap so it doesn't expand the view beyond the city.
    philly_boundary.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=2.5, zorder=2)
    ax.set_xlim(ax.get_xlim())
    ax.set_ylim(ax.get_ylim())
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zoom=12)

    # White halo behind each site type improves contrast against the basemap.
    for gsi_type, fill, edge in [("Basin", BASIN_COLOR, BASIN_EDGE), ("Bioinfiltration", BIO_COLOR, BIO_EDGE)]:
        subset = gdf_matched[gdf_matched["GSI Type"] == gsi_type]
        subset.plot(ax=ax, facecolor="none", edgecolor="white", linewidth=6, zorder=3)
        subset.plot(ax=ax, facecolor=fill, edgecolor=edge, linewidth=2.5, alpha=0.95, zorder=4)

    # Plotting can nudge the axis limits; restore the locked extent.
    ax.set_xlim(ax.get_xlim())
    ax.set_ylim(ax.get_ylim())

    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{mercator_to_lon(x):.3f}\u00b0"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{mercator_to_lat(y):.3f}\u00b0"))
    ax.tick_params(axis="both", labelsize=10, length=6, width=1.5, colors="black")
    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1.5)

    ax.set_xlabel("Longitude (\u00b0)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Latitude (\u00b0)", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.5, color="gray")

    n_basin = (gdf_matched["GSI Type"] == "Basin").sum()
    n_bio = (gdf_matched["GSI Type"] == "Bioinfiltration").sum()
    legend_elements = [
        Patch(facecolor=BASIN_COLOR, edgecolor=BASIN_EDGE, linewidth=2.5, label=f"Basin (n={n_basin})"),
        Patch(facecolor=BIO_COLOR, edgecolor=BIO_EDGE, linewidth=2.5, label=f"Bioinfiltration (n={n_bio})"),
        Patch(facecolor="none", edgecolor="black", linewidth=2.5, label="Philadelphia Boundary"),
    ]
    ax.legend(handles=legend_elements, fontsize=13, loc="upper right", frameon=True, framealpha=0.9, edgecolor="black", fancybox=False)

    plt.title("Spatial Distribution of GSI Sites in Philadelphia", fontsize=15, fontweight="bold", pad=15)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=350, bbox_inches="tight")
    plt.show()
    print(f"Saved: {output_path}")


def main():
    gdf_matched = load_site_geometries(POLYGON_SHAPEFILE, SITE_DATA_CSV)
    philly_boundary = load_city_boundary(CITY_NAME, CENSUS_PLACES_URL)
    plot_site_distribution(gdf_matched, philly_boundary, OUTPUT_PATH)


if __name__ == "__main__":
    main()
