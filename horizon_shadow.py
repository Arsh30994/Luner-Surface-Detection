"""Terrain illumination and doubly-shadowed crater detection.

The routines here are deterministic, NumPy-only implementations intended for
the hackathon demo pipeline:

1. Build or load a DEM.
2. Ray-march local horizons at multiple solar azimuths.
3. Mark permanently shadowed pixels as those shadowed for every azimuth.
4. Estimate secondary illumination from nearby lit terrain.
5. Extract connected doubly-shadowed crater-floor candidates.

The model is deliberately compact, but it is not a placeholder: shadowing is
computed from terrain elevation angles along solar rays.
"""

from __future__ import annotations

from collections import deque
from math import ceil
from typing import Iterable

import numpy as np


def _validate_dem(dem: np.ndarray) -> np.ndarray:
    arr = np.asarray(dem, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("DEM must be a 2-D array.")
    if min(arr.shape) < 8:
        raise ValueError("DEM must be at least 8 x 8 pixels.")
    if not np.isfinite(arr).all():
        raise ValueError("DEM contains NaN or infinite values.")
    return arr


def _box_mean(arr: np.ndarray, radius_px: int) -> np.ndarray:
    """Fast rectangular mean filter using an integral image."""
    arr = np.asarray(arr, dtype=np.float32)
    if radius_px <= 0:
        return arr.astype(np.float32, copy=True)

    h, w = arr.shape
    radius_px = int(radius_px)
    integral = np.pad(arr, ((1, 0), (1, 0)), mode="constant")
    integral = integral.cumsum(axis=0).cumsum(axis=1)

    rows = np.arange(h)
    cols = np.arange(w)
    r0 = np.maximum(0, rows - radius_px)
    r1 = np.minimum(h, rows + radius_px + 1)
    c0 = np.maximum(0, cols - radius_px)
    c1 = np.minimum(w, cols + radius_px + 1)

    sums = (
        integral[r1[:, None], c1[None, :]]
        - integral[r0[:, None], c1[None, :]]
        - integral[r1[:, None], c0[None, :]]
        + integral[r0[:, None], c0[None, :]]
    )
    counts = (r1 - r0)[:, None] * (c1 - c0)[None, :]
    return (sums / np.maximum(counts, 1)).astype(np.float32)


def _distance_to_sources(source_mask: np.ndarray) -> np.ndarray:
    """Approximate Euclidean distance to nearest source using chamfer passes."""
    source = np.asarray(source_mask, dtype=bool)
    h, w = source.shape
    inf = float(h + w + 1)
    dist = np.where(source, 0.0, inf).astype(np.float32)
    diag = np.float32(np.sqrt(2.0))

    for r in range(h):
        for c in range(w):
            best = dist[r, c]
            if r > 0:
                best = min(best, float(dist[r - 1, c] + 1.0))
                if c > 0:
                    best = min(best, float(dist[r - 1, c - 1] + diag))
                if c + 1 < w:
                    best = min(best, float(dist[r - 1, c + 1] + diag))
            if c > 0:
                best = min(best, float(dist[r, c - 1] + 1.0))
            dist[r, c] = best

    for r in range(h - 1, -1, -1):
        for c in range(w - 1, -1, -1):
            best = dist[r, c]
            if r + 1 < h:
                best = min(best, float(dist[r + 1, c] + 1.0))
                if c > 0:
                    best = min(best, float(dist[r + 1, c - 1] + diag))
                if c + 1 < w:
                    best = min(best, float(dist[r + 1, c + 1] + diag))
            if c + 1 < w:
                best = min(best, float(dist[r, c + 1] + 1.0))
            dist[r, c] = best

    return dist


def make_synthetic_dem(grid_size: int = 256, pixel_size_m: float = 10.0) -> np.ndarray:
    """Create a repeatable lunar south-pole style DEM with nested craters.

    The synthetic scene is useful for demos and tests when Chandrayaan-2 DEM
    products are not available locally. It contains raised rims, crater bowls,
    regional tilt, and smoothed regolith-scale roughness.
    """
    if grid_size < 64 or grid_size > 2048:
        raise ValueError("grid_size must be between 64 and 2048.")
    if pixel_size_m <= 0:
        raise ValueError("pixel_size_m must be positive.")

    n = int(grid_size)
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    x = (xx - n / 2) / n
    y = (yy - n / 2) / n

    rng = np.random.default_rng(20250629)
    dem = 18.0 * x - 8.0 * y
    dem += 1.8 * np.sin(2.4 * np.pi * x) + 1.2 * np.cos(2.1 * np.pi * y)

    rough = rng.normal(0.0, 1.0, (n, n)).astype(np.float32)
    rough = _box_mean(rough, max(1, n // 96))
    rough = rough / (float(np.std(rough)) + 1e-6)
    dem += rough * 1.5

    def add_crater(
        center_row: float,
        center_col: float,
        radius_px: float,
        depth_m: float,
        rim_m: float,
    ) -> None:
        nonlocal dem
        rr = np.hypot(yy - center_row, xx - center_col)
        rnorm = rr / max(radius_px, 1.0)

        inside = rnorm <= 1.0
        bowl = np.zeros_like(dem)
        bowl[inside] = -depth_m * (1.0 - rnorm[inside] ** 2) ** 0.55

        rim = rim_m * np.exp(-0.5 * ((rnorm - 1.0) / 0.085) ** 2)
        ejecta = 0.16 * rim_m * np.exp(-0.5 * ((rnorm - 1.25) / 0.22) ** 2)
        dem += bowl + rim + ejecta

    # A Faustini-like parent crater plus nested cold traps.
    add_crater(0.55 * n, 0.52 * n, 0.31 * n, 165.0, 35.0)
    add_crater(0.50 * n, 0.46 * n, 0.105 * n, 60.0, 18.0)
    add_crater(0.67 * n, 0.61 * n, 0.075 * n, 42.0, 12.0)
    add_crater(0.33 * n, 0.70 * n, 0.070 * n, 36.0, 10.0)

    # Keep map elevations centred and in metres.
    dem = dem.astype(np.float32)
    dem -= float(np.median(dem))
    return dem


def _shift_for_ray(arr: np.ndarray, dr: int, dc: int) -> np.ndarray:
    """Return arr sampled at [row + dr, col + dc], outside cells as NaN."""
    h, w = arr.shape
    out = np.full((h, w), np.nan, dtype=np.float32)

    if dr >= 0:
        src_r0, src_r1 = dr, h
        dst_r0, dst_r1 = 0, h - dr
    else:
        src_r0, src_r1 = 0, h + dr
        dst_r0, dst_r1 = -dr, h

    if dc >= 0:
        src_c0, src_c1 = dc, w
        dst_c0, dst_c1 = 0, w - dc
    else:
        src_c0, src_c1 = 0, w + dc
        dst_c0, dst_c1 = -dc, w

    if src_r1 > src_r0 and src_c1 > src_c0:
        out[dst_r0:dst_r1, dst_c0:dst_c1] = arr[src_r0:src_r1, src_c0:src_c1]
    return out


def _ray_offsets(theta: float, max_radius_px: int) -> Iterable[tuple[int, int, float]]:
    """Integer offsets along one ray, with adaptive spacing at long range."""
    seen: set[tuple[int, int]] = set()
    step = 1
    while step <= max_radius_px:
        dr = int(round(np.sin(theta) * step))
        dc = int(round(np.cos(theta) * step))
        if (dr, dc) != (0, 0) and (dr, dc) not in seen:
            seen.add((dr, dc))
            yield dr, dc, float(np.hypot(dr, dc))

        if step < 96:
            step += 1
        elif step < 192:
            step += 2
        else:
            step += 4


def compute_psr_fast(
    dem: np.ndarray,
    pixel_size_m: float = 10.0,
    sun_elevation_deg: float = 1.5,
    n_azimuths: int = 12,
    secondary_illum_threshold: float = 0.05,
    secondary_search_radius_m: float = 500.0,
) -> dict[str, np.ndarray]:
    """Compute PSR and doubly-shadowed masks from a DEM.

    A pixel is shadowed for one azimuth when any terrain sample along the ray
    to the sun rises above the solar elevation angle as seen from that pixel.
    Permanent shadow is the intersection of all sampled azimuth shadows.
    """
    dem = _validate_dem(dem)
    if pixel_size_m <= 0:
        raise ValueError("pixel_size_m must be positive.")
    if not (0.0 < sun_elevation_deg < 45.0):
        raise ValueError("sun_elevation_deg must be in (0, 45).")
    if n_azimuths < 4 or n_azimuths > 72:
        raise ValueError("n_azimuths must be between 4 and 72.")
    if secondary_search_radius_m <= 0:
        raise ValueError("secondary_search_radius_m must be positive.")

    h, w = dem.shape
    horizon_radius_px = min(
        max(h, w),
        max(64, int(ceil(3000.0 / pixel_size_m))),
    )
    tan_sun = float(np.tan(np.deg2rad(sun_elevation_deg)))
    lit_count = np.zeros((h, w), dtype=np.uint16)

    for theta in np.linspace(0.0, 2.0 * np.pi, int(n_azimuths), endpoint=False):
        max_apparent_slope = np.full((h, w), -np.inf, dtype=np.float32)
        for dr, dc, dist_px in _ray_offsets(theta, horizon_radius_px):
            shifted = _shift_for_ray(dem, dr, dc)
            apparent = (shifted - dem) / (dist_px * pixel_size_m)
            valid = np.isfinite(apparent)
            max_apparent_slope[valid] = np.maximum(max_apparent_slope[valid], apparent[valid])

        lit = max_apparent_slope < tan_sun
        lit_count += lit.astype(np.uint16)

    lit_fraction = (lit_count.astype(np.float32) / float(n_azimuths)).astype(np.float32)
    psr_mask = lit_count == 0

    search_radius_px = max(1, int(round(secondary_search_radius_m / pixel_size_m)))
    local_lit_fraction = _box_mean(lit_fraction, search_radius_px)
    distance_to_lit_px = _distance_to_sources(lit_fraction > 0.0)
    leakage_scale_px = max(1.0, 0.35 * search_radius_px)
    secondary_illum = local_lit_fraction * np.exp(-distance_to_lit_px / leakage_scale_px)
    doubly_shadowed_mask = psr_mask & (secondary_illum <= float(secondary_illum_threshold))

    return {
        "psr_mask": psr_mask,
        "doubly_shadowed_mask": doubly_shadowed_mask,
        "lit_fraction": lit_fraction,
        "secondary_illum": secondary_illum.astype(np.float32),
    }


def _connected_components(mask: np.ndarray, min_pixels: int) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []
    neighbors = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]

    starts = np.argwhere(mask)
    for sr, sc in starts:
        if visited[sr, sc]:
            continue
        q: deque[tuple[int, int]] = deque([(int(sr), int(sc))])
        visited[sr, sc] = True
        pixels: list[tuple[int, int]] = []

        while q:
            r, c = q.popleft()
            pixels.append((r, c))
            for dr, dc in neighbors:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not visited[nr, nc]:
                    visited[nr, nc] = True
                    q.append((nr, nc))

        if len(pixels) >= min_pixels:
            components.append(np.asarray(pixels, dtype=np.int32))

    return components


def detect_crater_candidates(
    doubly_shadowed_mask: np.ndarray,
    dem: np.ndarray,
    pixel_size_m: float,
    min_area_m2: float | None = None,
    max_candidates: int = 25,
) -> list[dict]:
    """Extract crater-floor candidates from connected doubly-shadowed regions."""
    dem = _validate_dem(dem)
    mask = np.asarray(doubly_shadowed_mask, dtype=bool)
    if mask.shape != dem.shape:
        raise ValueError("doubly_shadowed_mask shape must match DEM shape.")
    if pixel_size_m <= 0:
        raise ValueError("pixel_size_m must be positive.")

    min_area = min_area_m2 if min_area_m2 is not None else max(4.0 * pixel_size_m ** 2, 750.0)
    min_pixels = max(3, int(ceil(min_area / (pixel_size_m ** 2))))
    components = _connected_components(mask, min_pixels=min_pixels)
    yy, xx = np.mgrid[0:dem.shape[0], 0:dem.shape[1]]

    candidates: list[dict] = []
    for comp in components:
        rows = comp[:, 0].astype(np.float32)
        cols = comp[:, 1].astype(np.float32)
        n_px = int(len(comp))
        area_m2 = n_px * pixel_size_m ** 2
        centroid_row = float(rows.mean())
        centroid_col = float(cols.mean())
        radius_px = float(np.sqrt(n_px / np.pi))

        d2 = (yy - centroid_row) ** 2 + (xx - centroid_col) ** 2
        ring = (d2 >= (radius_px * 1.25) ** 2) & (d2 <= (radius_px * 2.0 + 2.0) ** 2)
        if ring.any():
            rim_elev = float(np.percentile(dem[ring], 70))
        else:
            rim_elev = float(np.percentile(dem, 75))
        floor_elev = float(np.percentile(dem[comp[:, 0], comp[:, 1]], 30))
        depth_m = max(0.0, rim_elev - floor_elev)

        coords = np.column_stack((rows - centroid_row, cols - centroid_col))
        if n_px >= 3:
            cov = np.cov(coords, rowvar=False)
            eig = np.linalg.eigvalsh(cov)
            roundness = float(np.sqrt(max(eig[0], 1e-6) / max(eig[-1], 1e-6)))
        else:
            roundness = 0.0

        area_score = float(np.clip(area_m2 / max(min_area * 12.0, 1.0), 0.0, 1.0))
        depth_score = float(np.clip(depth_m / 60.0, 0.0, 1.0))
        confidence = float(np.clip(0.40 * depth_score + 0.35 * roundness + 0.25 * area_score, 0.0, 1.0))

        candidates.append({
            "id": 0,
            "centroid_row": round(centroid_row, 2),
            "centroid_col": round(centroid_col, 2),
            "radius_px": round(radius_px, 2),
            "radius_m": round(radius_px * pixel_size_m, 1),
            "area_m2": round(area_m2, 1),
            "n_pixels": n_px,
            "depth_m": round(depth_m, 1),
            "confidence": round(confidence, 3),
        })

    candidates.sort(key=lambda c: (c["confidence"], c["area_m2"]), reverse=True)
    for idx, candidate in enumerate(candidates[:max_candidates], start=1):
        candidate["id"] = idx
    return candidates[:max_candidates]
