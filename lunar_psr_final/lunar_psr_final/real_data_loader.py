"""DEM loading helpers for local real-data runs.

Supported without extra dependencies:
- .npy  NumPy array
- .npz  first 2-D array in the archive, or key "dem" when present
- .csv/.txt whitespace or comma separated numeric grid

GeoTIFFs are supported when rasterio is installed in the runtime environment.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _validate_dem(arr: np.ndarray, source: Path) -> np.ndarray:
    dem = np.asarray(arr, dtype=np.float32)
    if dem.ndim != 2:
        raise ValueError(f"{source} did not contain a 2-D DEM grid.")
    if min(dem.shape) < 8:
        raise ValueError(f"{source} is too small for terrain analysis.")
    if not np.isfinite(dem).all():
        raise ValueError(f"{source} contains NaN or infinite values.")
    return dem


def load_dem(path: str | None) -> np.ndarray:
    if not path:
        raise ValueError("dem_path is required when dem_source is 'real'.")

    source = Path(path).expanduser()
    if not source.exists():
        raise FileNotFoundError(f"DEM file not found: {source}")

    suffix = source.suffix.lower()
    if suffix == ".npy":
        return _validate_dem(np.load(source), source)

    if suffix == ".npz":
        with np.load(source) as data:
            if "dem" in data:
                return _validate_dem(data["dem"], source)
            for key in data.files:
                arr = data[key]
                if np.asarray(arr).ndim == 2:
                    return _validate_dem(arr, source)
        raise ValueError(f"{source} did not contain a 2-D array.")

    if suffix in {".csv", ".txt", ".asc"}:
        try:
            arr = np.loadtxt(source, delimiter=",")
        except ValueError:
            arr = np.loadtxt(source)
        return _validate_dem(arr, source)

    if suffix in {".tif", ".tiff"}:
        try:
            import rasterio
        except ImportError as exc:
            raise ImportError("GeoTIFF DEM loading requires rasterio.") from exc
        with rasterio.open(source) as dataset:
            arr = dataset.read(1)
        return _validate_dem(arr, source)

    raise ValueError(f"Unsupported DEM format: {source.suffix}")
