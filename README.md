# GSI Hybrid OCSVM–XGBoost Framework

Code accompanying the manuscript *"Satellite-Based Condition Assessment of Urban Green Stormwater Infrastructure Using a Hybrid Machine Learning Framework."*

This repository contains the pipeline for processing PlanetScope imagery and Philadelphia Water Department (PWD) inspection data into site-level features and training the hybrid OCSVM–XGBoost classification framework, including ablation and SHAP analyses.

## Data Availability

This repository does not include:
- **PlanetScope imagery**, obtained through Planet Labs' Education and Research Program and not permitted for redistribution.
- **PWD inspection records**, provided by Philadelphia Water Department and not publicly available.

## Repository Structure

```
src/
├── data_acquisition/
│   └── planet_download.py          # Search, order, and download PlanetScope
│                                    # imagery clipped to GSI site boundaries
├── feature_engineering/
│   ├── extract_pixel_values.py     # Extract per-pixel band values from
│                                    # clipped raster files
│   ├── compute_spectral_indices.py # Compute NDVI, SAVI, GRVI, NDWI, BI,
│                                    # NDBI(modified) from reflectance bands
│   └── build_site_features.py      # Quantile-bin features and merge with
│                                    # inspection, rainfall, and geometry data
├── modeling/
│   ├── ocsvm_xgboost_pipeline.py   # Core hybrid model: OCSVM anomaly
│                                    # feature + SMOTE + XGBoost, with
│                                    # hyperparameter search and evaluation
│   ├── ablation_study.py           # Ablation comparison (XGBoost only /
│                                    # SMOTE + XGBoost / full hybrid) with
│                                    # paired significance testing, run
│                                    # across all five inspection targets
│   └── shap_analysis.py            # SHAP feature importance for a fitted
│                                    # hybrid model
└── visualization/
    └── performance_figures.py      # Summary bar charts of model
                                     # performance across inspection targets
```

## Setup

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your own Planet API key (required
only for `data_acquisition/planet_download.py`):

```bash
cp .env.example .env
```

## Usage

Run scripts in the following order:

1. planet_download.py — acquire PlanetScope imagery
2. extract_pixel_values.py — extract spectral values
3. compute_spectral_indices.py — calculate spectral indices
4. build_site_features.py — generate site-level features
5. ocsvm_xgboost_pipeline.py — train and evaluate the hybrid model
6. ablation_study.py — compare model variants
7. shap_analysis.py — generate feature importance results

Paths should be updated in each script configuration section before execution.

## Citation

If you use this code, please cite the associated manuscript (citation to
be added upon publication).
