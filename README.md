# Lunar PSR Pipeline — Combined Project

FastAPI + vanilla JS hackathon project for lunar south-pole terrain analysis:

1. PSR and doubly-shadowed crater detection from DEM horizon shadowing.
2. Landing-site scoring from slope, roughness/boulder density, and ellipse checks.
3. Rover traverse planning with A* over terrain cost and a reachable access-point fallback.
4. Ice-volume estimation from doubly-shadowed area, radar depth, concentration, and Monte Carlo uncertainty.

## Project structure

```
main.py, horizon_shadow.py, ice_volume.py,   ← FastAPI backend + pipeline modules
landing_site.py, rover_path.py,
real_data_loader.py, requirements.txt
tests/                                       ← unit + HTTP integration tests
web/
  index.html                                 ← overview / pitch page (design, formulas, rubric)
  dashboard.html                              ← live working demo (calls the API, renders results)
```

`index.html` is the polished front door judges see first — it explains the pipeline, radar
thresholds, and rubric alignment. Its "Launch Live Demo" button and nav link take you to
`dashboard.html`, the real interactive tool that talks to the FastAPI backend. Both pages
share the same font/design system (Space Grotesk, Space Mono, Inter) so they read as one
product.

## Run (single command)

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Then open **http://localhost:8000/** — this serves `web/index.html` at the root and
`web/dashboard.html` at `/dashboard.html`, with the API available under `/api/*`.

## Run (opening the HTML files directly)

You can still just open `web/index.html` or `web/dashboard.html` straight in a browser
without uvicorn serving them — `dashboard.html`'s JS talks to `http://localhost:8000`
directly (CORS is enabled), so just make sure `uvicorn main:app` is running separately.

## Test

```bash
python -m unittest discover -s tests
```

The tests run the full synthetic pipeline end to end without FastAPI.

## Real DEM Input

Set `dem_source` to `real` and provide `dem_path`. Supported formats without extra
dependencies are `.npy`, `.npz`, `.csv`, `.txt`, and `.asc`. GeoTIFF `.tif/.tiff` works
when `rasterio` is installed.

## Note on `index.html`

A few sections of the overview page (the "live" radar classifier, the procedural slope
map, and the animated ice-volume bars) currently use illustrative/simulated numbers for
presentation purposes rather than calling the real backend. If you want those wired to
live API results instead, that's a separate follow-up — ask and I can point to exactly
which functions to swap out.
