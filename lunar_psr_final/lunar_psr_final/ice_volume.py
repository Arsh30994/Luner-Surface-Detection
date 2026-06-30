"""Step 5: ice-volume estimation for doubly-shadowed lunar cold traps.

Volume is estimated as:

    water_ice_volume = area * sampled_depth * sampled_concentration

The concentration map is derived from the secondary-illumination field produced
by the PSR pipeline: lower local illumination means stronger thermal isolation,
so the pixel is assigned a concentration closer to conc_max. Monte Carlo draws
sample depth and concentration uncertainty, producing p5/p50/p95 intervals.

This module is pure NumPy and deterministic under a fixed seed.
"""

from __future__ import annotations

import numpy as np


DEFAULT_RADAR_DEPTH_M = 5.0
DEFAULT_DEPTH_SIGMA_M = 1.0
DEFAULT_CONC_MIN = 0.10
DEFAULT_CONC_MAX = 0.20
DEFAULT_CONC_SIGMA = 0.04
N_MC = 5_000


def _validate_inputs(
    doubly_shadowed_mask: np.ndarray,
    secondary_illum: np.ndarray,
    pixel_size_m: float,
    radar_depth_m: float,
    depth_sigma_m: float,
    conc_min: float,
    conc_max: float,
    conc_sigma: float,
    n_mc: int,
) -> tuple[np.ndarray, np.ndarray]:
    mask = np.asarray(doubly_shadowed_mask, dtype=bool)
    illum = np.asarray(secondary_illum, dtype=np.float32)
    if mask.ndim != 2 or illum.ndim != 2:
        raise ValueError("doubly_shadowed_mask and secondary_illum must be 2-D arrays.")
    if mask.shape != illum.shape:
        raise ValueError("doubly_shadowed_mask and secondary_illum shapes must match.")
    if not np.isfinite(illum).all():
        raise ValueError("secondary_illum contains NaN or infinite values.")
    if pixel_size_m <= 0:
        raise ValueError("pixel_size_m must be positive.")
    if radar_depth_m <= 0:
        raise ValueError("radar_depth_m must be positive.")
    if depth_sigma_m < 0:
        raise ValueError("depth_sigma_m cannot be negative.")
    if not (0.0 <= conc_min <= conc_max <= 1.0):
        raise ValueError("concentration bounds must satisfy 0 <= conc_min <= conc_max <= 1.")
    if conc_sigma < 0:
        raise ValueError("conc_sigma cannot be negative.")
    if n_mc < 100:
        raise ValueError("n_mc must be at least 100 for stable uncertainty intervals.")
    return mask, illum


def _robust_norm01(values: np.ndarray) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float32)
    if vals.size == 0:
        return vals
    lo, hi = np.percentile(vals, [5, 95])
    if hi <= lo + 1e-9:
        return np.full(vals.shape, 0.5, dtype=np.float32)
    return np.clip((vals - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _concentration_map(
    secondary_illum: np.ndarray,
    mask: np.ndarray,
    conc_min: float = DEFAULT_CONC_MIN,
    conc_max: float = DEFAULT_CONC_MAX,
) -> np.ndarray:
    """Map lower secondary illumination to higher water-ice concentration."""
    conc = np.zeros(secondary_illum.shape, dtype=np.float32)
    if not mask.any():
        return conc

    illum_vals = secondary_illum[mask]
    isolation = 1.0 - _robust_norm01(illum_vals)
    conc[mask] = conc_min + (conc_max - conc_min) * isolation
    return conc.astype(np.float32)


def _mc_volume(
    area_m2: float,
    mean_conc: float,
    radar_depth_m: float = DEFAULT_RADAR_DEPTH_M,
    depth_sigma_m: float = DEFAULT_DEPTH_SIGMA_M,
    conc_sigma: float = DEFAULT_CONC_SIGMA,
    n_mc: int = N_MC,
    rng: np.random.Generator | None = None,
) -> dict:
    if rng is None:
        rng = np.random.default_rng(0)
    if area_m2 <= 0 or mean_conc <= 0:
        return {
            "vol_m3_p5": 0.0,
            "vol_m3_p50": 0.0,
            "vol_m3_p95": 0.0,
            "vol_km3_p5": 0.0,
            "vol_km3_p50": 0.0,
            "vol_km3_p95": 0.0,
        }

    depths = rng.normal(radar_depth_m, depth_sigma_m, int(n_mc))
    depths = np.clip(depths, max(0.05, radar_depth_m * 0.05), radar_depth_m * 3.0)

    concs = rng.normal(mean_conc, conc_sigma, int(n_mc))
    concs = np.clip(concs, 0.0, 1.0)
    volumes = area_m2 * depths * concs

    p5, p50, p95 = np.percentile(volumes, [5, 50, 95])
    return {
        "vol_m3_p5": round(float(p5), 1),
        "vol_m3_p50": round(float(p50), 1),
        "vol_m3_p95": round(float(p95), 1),
        "vol_km3_p5": round(float(p5) * 1e-9, 6),
        "vol_km3_p50": round(float(p50) * 1e-9, 6),
        "vol_km3_p95": round(float(p95) * 1e-9, 6),
    }


def _pixel_assignments(mask: np.ndarray, craters: list[dict]) -> dict[object, np.ndarray]:
    """Assign each ice pixel to one crater, avoiding per-crater double counts."""
    assignments: dict[object, np.ndarray] = {}
    if not mask.any():
        return assignments

    rows, cols = np.where(mask)
    if not craters:
        all_mask = np.zeros(mask.shape, dtype=bool)
        all_mask[rows, cols] = True
        assignments["all"] = all_mask
        return assignments

    norms = []
    crater_ids = []
    for crater in craters:
        cr = float(crater["centroid_row"])
        cc = float(crater["centroid_col"])
        rp = max(float(crater.get("radius_px", 1.0)), 1.0)
        norms.append(np.hypot(rows - cr, cols - cc) / rp)
        crater_ids.append(crater["id"])
    norm_stack = np.vstack(norms)
    best_idx = np.argmin(norm_stack, axis=0)
    best_norm = norm_stack[best_idx, np.arange(rows.size)]

    for idx, crater_id in enumerate(crater_ids):
        pick = (best_idx == idx) & (best_norm <= 1.75)
        if not np.any(pick):
            continue
        crater_mask = np.zeros(mask.shape, dtype=bool)
        crater_mask[rows[pick], cols[pick]] = True
        assignments[crater_id] = crater_mask

    leftover = best_norm > 1.75
    if np.any(leftover):
        leftover_mask = np.zeros(mask.shape, dtype=bool)
        leftover_mask[rows[leftover], cols[leftover]] = True
        assignments["unassigned"] = leftover_mask

    return assignments


def _per_region_volume(
    region_id: object,
    region_mask: np.ndarray,
    conc_map: np.ndarray,
    pixel_size_m: float,
    radar_depth_m: float,
    depth_sigma_m: float,
    conc_sigma: float,
    n_mc: int,
    rng: np.random.Generator,
) -> dict:
    n_px = int(region_mask.sum())
    area_m2 = n_px * pixel_size_m ** 2
    mean_conc = float(conc_map[region_mask].mean()) if n_px else 0.0
    mc = _mc_volume(
        area_m2=area_m2,
        mean_conc=mean_conc,
        radar_depth_m=radar_depth_m,
        depth_sigma_m=depth_sigma_m,
        conc_sigma=conc_sigma,
        n_mc=n_mc,
        rng=rng,
    )
    return {
        "crater_id": region_id,
        "n_ice_pixels": n_px,
        "area_m2": round(area_m2, 1),
        "area_km2": round(area_m2 * 1e-6, 6),
        "mean_conc_pct": round(mean_conc * 100.0, 2),
        "depth_m": float(radar_depth_m),
        **mc,
    }


def estimate_ice_volume(
    doubly_shadowed_mask: np.ndarray,
    secondary_illum: np.ndarray,
    craters: list[dict],
    pixel_size_m: float,
    radar_depth_m: float = DEFAULT_RADAR_DEPTH_M,
    depth_sigma_m: float = DEFAULT_DEPTH_SIGMA_M,
    conc_min: float = DEFAULT_CONC_MIN,
    conc_max: float = DEFAULT_CONC_MAX,
    conc_sigma: float = DEFAULT_CONC_SIGMA,
    n_mc: int = N_MC,
    seed: int = 42,
) -> dict:
    """Estimate total and per-crater water-ice volume in the top depth slice."""
    mask, illum = _validate_inputs(
        doubly_shadowed_mask=doubly_shadowed_mask,
        secondary_illum=secondary_illum,
        pixel_size_m=pixel_size_m,
        radar_depth_m=radar_depth_m,
        depth_sigma_m=depth_sigma_m,
        conc_min=conc_min,
        conc_max=conc_max,
        conc_sigma=conc_sigma,
        n_mc=n_mc,
    )

    rng = np.random.default_rng(seed)
    total_px = int(mask.sum())
    area_per_px = pixel_size_m ** 2
    total_area_m2 = total_px * area_per_px
    conc_map = _concentration_map(illum, mask, conc_min=conc_min, conc_max=conc_max)
    mean_conc = float(conc_map[mask].mean()) if total_px else 0.0

    total_mc = _mc_volume(
        area_m2=total_area_m2,
        mean_conc=mean_conc,
        radar_depth_m=radar_depth_m,
        depth_sigma_m=depth_sigma_m,
        conc_sigma=conc_sigma,
        n_mc=n_mc,
        rng=rng,
    )

    assignments = _pixel_assignments(mask, craters)
    per_crater = [
        _per_region_volume(
            region_id=region_id,
            region_mask=region_mask,
            conc_map=conc_map,
            pixel_size_m=pixel_size_m,
            radar_depth_m=radar_depth_m,
            depth_sigma_m=depth_sigma_m,
            conc_sigma=conc_sigma,
            n_mc=n_mc,
            rng=rng,
        )
        for region_id, region_mask in assignments.items()
    ]
    per_crater.sort(key=lambda row: row["vol_m3_p50"], reverse=True)

    method_note = (
        f"Volume = area x depth x concentration. Area comes from {total_px} "
        f"doubly-shadowed pixels at {pixel_size_m:g} m/px. Concentration is "
        f"mapped from secondary-illumination isolation into {conc_min*100:.0f}-"
        f"{conc_max*100:.0f}% ice fraction. Uncertainty uses {n_mc:,} Monte "
        f"Carlo samples over depth {radar_depth_m:g}+/-{depth_sigma_m:g} m and "
        f"concentration sigma {conc_sigma*100:.1f} percentage points."
    )

    return {
        "total_area_m2": round(total_area_m2, 1),
        "total_area_km2": round(total_area_m2 * 1e-6, 6),
        "mean_conc_pct": round(mean_conc * 100.0, 2),
        "radar_depth_m": float(radar_depth_m),
        "depth_sigma_m": float(depth_sigma_m),
        "conc_min_pct": round(conc_min * 100.0, 2),
        "conc_max_pct": round(conc_max * 100.0, 2),
        "conc_sigma_pct": round(conc_sigma * 100.0, 2),
        "n_ice_pixels": total_px,
        "total_vol_m3_p5": total_mc["vol_m3_p5"],
        "total_vol_m3_p50": total_mc["vol_m3_p50"],
        "total_vol_m3_p95": total_mc["vol_m3_p95"],
        "total_vol_km3_p5": total_mc["vol_km3_p5"],
        "total_vol_km3_p50": total_mc["vol_km3_p50"],
        "total_vol_km3_p95": total_mc["vol_km3_p95"],
        "per_crater": per_crater,
        "conc_map": conc_map,
        "method_note": method_note,
    }
