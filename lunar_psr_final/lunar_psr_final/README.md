# Lunar PSR Pipeline - Step 5 Hardened

FastAPI + vanilla JS demo for a hackathon lunar south-pole workflow:

1. PSR and doubly-shadowed crater detection from DEM horizon shadowing.
2. Landing-site scoring from slope, roughness/boulder density, and ellipse checks.
3. Rover traverse planning with A* over terrain cost and a reachable access-point fallback.
4. Ice-volume estimation from doubly-shadowed area, radar depth, concentration, and Monte Carlo uncertainty.

## Run

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Then open `index.html` in a browser. The UI calls `http://localhost:8000`.

## Test

```bash
python -m unittest discover -s tests
```

The tests run the full synthetic pipeline end to end without FastAPI.

## Real DEM Input

Set `dem_source` to `real` and provide `dem_path`. Supported formats without extra dependencies are `.npy`, `.npz`, `.csv`, `.txt`, and `.asc`. GeoTIFF `.tif/.tiff` works when `rasterio` is installed.
