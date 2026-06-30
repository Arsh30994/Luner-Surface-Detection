"""Landing-site analysis from DEM-derived terrain constraints."""

from __future__ import annotations

import numpy as np

from horizon_shadow import _box_mean


def _validate_dem(dem: np.ndarray) -> np.ndarray:
    arr = np.asarray(dem, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("DEM must be a 2-D array.")
    if not np.isfinite(arr).all():
        raise ValueError("DEM contains NaN or infinite values.")
    return arr


def _terrain_slope_aspect(dem: np.ndarray, pixel_size_m: float) -> tuple[np.ndarray, np.ndarray]:
    gy, gx = np.gradient(dem, pixel_size_m, pixel_size_m)
    slope_rad = np.arctan(np.hypot(gx, gy))
    aspect = (np.degrees(np.arctan2(-gx, gy)) + 360.0) % 360.0
    return np.degrees(slope_rad).astype(np.float32), aspect.astype(np.float32)


def _synthetic_ohrc(dem: np.ndarray, slope_deg: np.ndarray, aspect_deg: np.ndarray) -> np.ndarray:
    """Build a deterministic panchromatic hillshade-like OHRC proxy."""
    sun_az = np.deg2rad(315.0)
    sun_el = np.deg2rad(25.0)
    slope = np.deg2rad(slope_deg)
    aspect = np.deg2rad(aspect_deg)

    shade = (
        np.sin(sun_el) * np.cos(slope)
        + np.cos(sun_el) * np.sin(slope) * np.cos(sun_az - aspect)
    )
    shade = np.clip(shade, 0.0, 1.0)

    local = dem - _box_mean(dem, 5)
    local = local / (float(np.percentile(np.abs(local), 98)) + 1e-6)
    img = 0.70 * shade + 0.30 * np.clip(0.5 + 0.5 * local, 0.0, 1.0)
    return np.clip(img, 0.0, 1.0).astype(np.float32)


def _boulder_density(dem: np.ndarray) -> np.ndarray:
    """Estimate boulder/roughness density from local high-pass relief."""
    smooth = _box_mean(dem, 3)
    high = dem - smooth
    rough = np.sqrt(_box_mean(high * high, 2))
    scale = float(np.percentile(rough, 97)) + 1e-6
    rough_norm = np.clip(rough / scale, 0.0, 1.0)

    lap = (
        np.roll(dem, 1, axis=0) + np.roll(dem, -1, axis=0)
        + np.roll(dem, 1, axis=1) + np.roll(dem, -1, axis=1)
        - 4.0 * dem
    )
    curv = np.abs(lap)
    curv = np.clip(curv / (float(np.percentile(curv, 97)) + 1e-6), 0.0, 1.0)

    density = 2.5 + 11.0 * rough_norm + 5.0 * curv
    return np.clip(density, 0.0, 30.0).astype(np.float32)


def _ellipse_offsets(semi_major_px: int, semi_minor_px: int) -> tuple[np.ndarray, np.ndarray]:
    a = max(1, int(semi_major_px))
    b = max(1, int(semi_minor_px))
    rr, cc = np.mgrid[-a:a + 1, -b:b + 1]
    mask = (rr / a) ** 2 + (cc / b) ** 2 <= 1.0
    return rr[mask].astype(np.int32), cc[mask].astype(np.int32)


def _nearest_crater(row: int, col: int, craters: list[dict]) -> tuple[dict | None, float, float]:
    if not craters:
        return None, float("inf"), float("inf")
    best = None
    best_center_dist_px = float("inf")
    best_edge_dist_px = float("inf")
    for crater in craters:
        dr = row - float(crater["centroid_row"])
        dc = col - float(crater["centroid_col"])
        center_dist = float(np.hypot(dr, dc))
        edge_dist = center_dist - float(crater.get("radius_px", 0.0))
        if center_dist < best_center_dist_px:
            best = crater
            best_center_dist_px = center_dist
            best_edge_dist_px = edge_dist
    return best, best_center_dist_px, best_edge_dist_px


def _find_landing_sites(
    safety_score: np.ndarray,
    hard_unsafe: np.ndarray,
    craters: list[dict],
    pixel_size_m: float,
    ellipse_semi_major_m: float,
    ellipse_semi_minor_m: float,
    top_k_sites: int,
) -> list[dict]:
    h, w = safety_score.shape
    a_px = max(2, int(round(ellipse_semi_major_m / pixel_size_m)))
    b_px = max(2, int(round(ellipse_semi_minor_m / pixel_size_m)))
    off_r, off_c = _ellipse_offsets(a_px, b_px)

    margin = max(a_px, b_px) + 2
    if margin * 2 >= min(h, w):
        raise ValueError("Landing ellipse is too large for the DEM grid.")

    stride = max(2, min(a_px, b_px) // 2)
    desired_edge_m = max(350.0, ellipse_semi_major_m * 1.8)
    sigma_edge_m = max(250.0, desired_edge_m * 0.55)

    candidates: list[dict] = []
    for row in range(margin, h - margin, stride):
        rr = row + off_r
        for col in range(margin, w - margin, stride):
            cc = col + off_c
            vals = safety_score[rr, cc]
            unsafe_frac = float(hard_unsafe[rr, cc].mean())
            if unsafe_frac > 0.04:
                continue

            mean_score = float(vals.mean())
            min_score = float(vals.min())
            if mean_score < 48.0 or min_score < 14.0:
                continue

            crater, center_dist_px, edge_dist_px = _nearest_crater(row, col, craters)
            edge_dist_m = edge_dist_px * pixel_size_m
            if crater is not None and edge_dist_m < ellipse_semi_major_m * 0.7:
                continue

            if crater is None:
                proximity_bonus = 0.0
            else:
                proximity_bonus = 16.0 * float(
                    np.exp(-0.5 * ((edge_dist_m - desired_edge_m) / sigma_edge_m) ** 2)
                )

            rank_score = mean_score + proximity_bonus + 0.18 * min_score - 85.0 * unsafe_frac
            candidates.append({
                "rank_score": rank_score,
                "centroid_row": int(row),
                "centroid_col": int(col),
                "mean_score": round(mean_score, 2),
                "min_score": round(min_score, 2),
                "unsafe_fraction": round(unsafe_frac, 4),
                "semi_major_m": float(ellipse_semi_major_m),
                "semi_minor_m": float(ellipse_semi_minor_m),
                "nearest_crater_id": crater["id"] if crater else None,
                "distance_to_target_m": round(center_dist_px * pixel_size_m, 1)
                if crater else None,
                "edge_clearance_m": round(edge_dist_m, 1) if crater else None,
            })

    candidates.sort(key=lambda item: item["rank_score"], reverse=True)

    # Non-maximum suppression so the top list represents distinct ellipses.
    chosen: list[dict] = []
    min_sep_px = max(a_px, b_px)
    for cand in candidates:
        if len(chosen) >= top_k_sites:
            break
        if all(
            np.hypot(cand["centroid_row"] - s["centroid_row"], cand["centroid_col"] - s["centroid_col"])
            >= min_sep_px
            for s in chosen
        ):
            cand = dict(cand)
            cand.pop("rank_score", None)
            cand["id"] = len(chosen) + 1
            chosen.append(cand)

    return chosen


def run_landing_site_analysis(
    dem: np.ndarray,
    pixel_size_m: float,
    craters: list[dict],
    slope_redline_deg: float = 15.0,
    slope_caution_deg: float = 5.0,
    boulder_redline: float = 12.0,
    ellipse_semi_major_m: float = 150.0,
    ellipse_semi_minor_m: float = 100.0,
    top_k_sites: int = 5,
) -> dict:
    """Score terrain safety and identify the best landing ellipses."""
    dem = _validate_dem(dem)
    if pixel_size_m <= 0:
        raise ValueError("pixel_size_m must be positive.")
    if not (0 <= slope_caution_deg < slope_redline_deg):
        raise ValueError("slope_caution_deg must be lower than slope_redline_deg.")
    if boulder_redline <= 0:
        raise ValueError("boulder_redline must be positive.")
    if ellipse_semi_major_m <= 0 or ellipse_semi_minor_m <= 0:
        raise ValueError("Landing ellipse dimensions must be positive.")
    if top_k_sites <= 0:
        raise ValueError("top_k_sites must be positive.")

    slope_deg, aspect_deg = _terrain_slope_aspect(dem, pixel_size_m)
    ohrc_img = _synthetic_ohrc(dem, slope_deg, aspect_deg)
    boulder_density = _boulder_density(dem)

    slope_redflag = slope_deg >= slope_redline_deg
    boulder_redflag = boulder_density >= boulder_redline
    hard_unsafe = slope_redflag | boulder_redflag

    slope_span = max(slope_redline_deg - slope_caution_deg, 1e-6)
    slope_score = 1.0 - np.clip((slope_deg - slope_caution_deg) / slope_span, 0.0, 1.0)
    boulder_score = 1.0 - np.clip(boulder_density / boulder_redline, 0.0, 1.0)
    safety_score = 100.0 * (0.68 * slope_score + 0.32 * boulder_score)
    safety_score = np.where(hard_unsafe, np.minimum(safety_score, 18.0), safety_score)
    safety_score = np.clip(safety_score, 0.0, 100.0).astype(np.float32)

    landing_sites = _find_landing_sites(
        safety_score=safety_score,
        hard_unsafe=hard_unsafe,
        craters=craters,
        pixel_size_m=pixel_size_m,
        ellipse_semi_major_m=ellipse_semi_major_m,
        ellipse_semi_minor_m=ellipse_semi_minor_m,
        top_k_sites=top_k_sites,
    )

    return {
        "slope_deg": slope_deg,
        "aspect_deg": aspect_deg,
        "slope_redflag": slope_redflag,
        "ohrc_img": ohrc_img,
        "boulder_density": boulder_density,
        "safety_score": safety_score,
        "hard_unsafe": hard_unsafe,
        "landing_sites": landing_sites,
        "boulder_count": int(np.count_nonzero(boulder_redflag)),
        "pct_terrain_safe": round(float((~hard_unsafe).mean() * 100.0), 2),
        "pct_slope_redflag": round(float(slope_redflag.mean() * 100.0), 2),
    }
