"""
Search Planet’s Data API for PlanetScope scenes matching each GSI site inspection date, 
order clipped analytic surface reflectance imagery, and download the resulting GeoTIFFs.

Requires a Planet Labs API key with access to the analytic_sr_udm2 product.
"""

import asyncio
import os
from datetime import datetime

import aiohttp
import geopandas as gpd
import nest_asyncio
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
nest_asyncio.apply()

# --- CONFIG ---
PLANET_API_KEY = os.environ["PLANET_API_KEY"]
ORDERS_URL = "https://api.planet.com/compute/ops/orders/v2"
SEARCH_URL = "https://api.planet.com/data/v1/quick-search"
ITEM_TYPE = "PSScene"
PRODUCT_BUNDLE = "analytic_sr_udm2"

POLYGON_SHAPEFILE = "data/filtered_gsi_polygons.shp"
SITE_MATCH_CSV = "data/gsi_planet_final_results.csv"  # SMPId, image_id columns
OUTPUT_DIR = "data/dataWithExtractedBands"

# Row range within SITE_MATCH_CSV to process in this run. Processing in
# batches makes it easier to resume after an interrupted run and keeps
# individual Colab sessions within a manageable runtime.
START_ROW = 0
END_ROW = 50

CLOUD_COVER_THRESHOLD_PCT = 10
ORDER_POLL_INTERVAL_SEC = 10
ORDER_MAX_POLL_ATTEMPTS = 120  # ~20 minutes


def build_geojson_geometry(geometry):
    """Convert a shapely Polygon/MultiPolygon to a GeoJSON geometry dict."""
    if geometry.geom_type == "Polygon":
        return {"type": "Polygon", "coordinates": [list(geometry.exterior.coords)]}
    if geometry.geom_type == "MultiPolygon":
        return {
            "type": "MultiPolygon",
            "coordinates": [[list(p.exterior.coords)] for p in geometry.geoms],
        }
    raise ValueError(f"Unsupported geometry type: {geometry.geom_type}")


def find_best_scene(geometry, inspection_date, cloud_threshold_pct=CLOUD_COVER_THRESHOLD_PCT):
    """
    Search for the least-cloudy PlanetScope scene covering `geometry`,
    preferring the inspection date itself and falling back to +/-1 day.
    """
    import requests

    if pd.isna(inspection_date):
        return {"status": "No Inspection Date", "image_id": None}

    geom = build_geojson_geometry(geometry)

    for date_offset in (0, 1, -1):
        target_date = inspection_date + pd.Timedelta(days=date_offset)
        start = target_date.strftime("%Y-%m-%dT00:00:00.000Z")
        end = target_date.strftime("%Y-%m-%dT23:59:59.999Z")

        payload = {
            "item_types": [ITEM_TYPE],
            "filter": {
                "type": "AndFilter",
                "config": [
                    {"type": "GeometryFilter", "field_name": "geometry", "config": geom},
                    {
                        "type": "DateRangeFilter",
                        "field_name": "acquired",
                        "config": {"gte": start, "lte": end},
                    },
                    {
                        "type": "RangeFilter",
                        "field_name": "cloud_cover",
                        "config": {"lte": cloud_threshold_pct / 100.0},
                    },
                ],
            },
        }

        response = requests.post(SEARCH_URL, auth=(PLANET_API_KEY, ""), json=payload, timeout=30)
        response.raise_for_status()
        features = response.json().get("features", [])

        if features:
            features.sort(key=lambda f: f["properties"].get("cloud_cover", 1.0))
            best = features[0]
            return {
                "status": "Match Found",
                "image_id": best["id"],
                "acquisition_date": best["properties"]["acquired"][:10],
                "cloud_cover": round(best["properties"].get("cloud_cover", 0) * 100, 2),
                "date_difference": date_offset,
            }

    return {"status": "No Match (Cloud or Availability)", "image_id": None}


async def create_and_download_order(geometry, image_id, smp_id, session, output_dir):
    """Place a clipped order for one scene and download the resulting image."""
    order_request = {
        "name": f"GSI_{smp_id}_{image_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "products": [{"item_ids": [image_id], "item_type": ITEM_TYPE, "product_bundle": PRODUCT_BUNDLE}],
        "tools": [{"clip": {"aoi": geometry.__geo_interface__}}],
    }
    headers = {"Content-Type": "application/json", "Authorization": f"api-key {PLANET_API_KEY}"}

    async with session.post(ORDERS_URL, json=order_request, headers=headers) as resp:
        if resp.status not in (200, 202):
            print(f"  Order creation failed ({resp.status}): {await resp.text()}")
            return None
        order = await resp.json()

    order_id = order["id"]
    order_url = f"{ORDERS_URL}/{order_id}"

    for attempt in range(ORDER_MAX_POLL_ATTEMPTS):
        async with session.get(order_url, headers=headers) as resp:
            status = await resp.json()
        state = status["state"]

        if state == "success":
            break
        if state in ("failed", "partial"):
            print(f"  Order failed: {status.get('error', 'unknown error')}")
            return None
        await asyncio.sleep(ORDER_POLL_INTERVAL_SEC)
    else:
        print(f"  Order timed out after {ORDER_MAX_POLL_ATTEMPTS * ORDER_POLL_INTERVAL_SEC}s")
        return None

    download_url = next(
        (r["location"] for r in status["_links"]["results"] if "AnalyticMS_SR" in r["name"]), None
    )
    if not download_url:
        print("  No analytic surface reflectance product found in order results")
        return None

    async with session.get(download_url) as resp:
        if resp.status != 200:
            print(f"  Download failed with status {resp.status}")
            return None
        image_data = await resp.read()

    output_path = os.path.join(output_dir, f"SMP_{smp_id}_{image_id}.tif")
    with open(output_path, "wb") as f:
        f.write(image_data)
    print(f"  Saved: {output_path}")
    return output_path


async def process_batch(start_row, end_row):
    gdf = gpd.read_file(POLYGON_SHAPEFILE)
    df = pd.read_csv(SITE_MATCH_CSV)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    subset = df.iloc[start_row:end_row]
    print(f"Processing rows {start_row}-{end_row - 1} ({len(subset)} sites)")

    async with aiohttp.ClientSession() as session:
        for idx, row in subset.iterrows():
            smp_id, image_id = row["SMPId"], row["image_id"]
            print(f"[row {idx}] SMPId {smp_id}, image {image_id}")

            geom_rows = gdf[gdf["SMPId"] == smp_id]
            if geom_rows.empty:
                print("  No matching geometry, skipping")
                continue

            await create_and_download_order(geom_rows.geometry.iloc[0], image_id, smp_id, session, OUTPUT_DIR)


if __name__ == "__main__":
    asyncio.run(process_batch(START_ROW, END_ROW))
