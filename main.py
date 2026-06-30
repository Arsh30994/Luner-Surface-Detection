"""
FastAPI backend for lunar PSR / doubly-shadowed crater detection.

Endpoints:
  POST /api/analyze      – start a detection job (returns job_id)
  GET  /api/status/{id}  – poll job status + progress
  GET  /api/result/{id}  – full JSON result with crater list
  GET  /api/layer/{id}/{name}.png  – rendered PNG layer for the map
  GET  /api/dem/{id}/profile       – elevation profile across DEM
"""

import asyncio
import io
import time
import uuid
from typing import Any

import numpy as np
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field, model_validator

try:
    from PIL import Image
except ImportError:
    Image = None  # will fall back to numpy PNG writer

from horizon_shadow import (
    make_synthetic_dem,
    compute_psr_fast,
    detect_crater_candidates,
)
from landing_site import run_landing_site_analysis
from rover_path import plan_rover_path
from ice_volume import estimate_ice_volume

app = FastAPI(title="Lunar PSR Detector", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory job store ───────────────────────────────────────────────────────
jobs: dict[str, dict[str, Any]] = {}


# ── Request / Response models ─────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    grid_size: int = Field(256, ge=64, le=2048)
    pixel_size_m: float = Field(10.0, gt=0)
    sun_elevation_deg: float = Field(1.5, gt=0, lt=45)
    n_azimuths: int = Field(12, ge=4, le=72)
    secondary_illum_threshold: float = Field(0.05, ge=0, le=1)
    secondary_search_radius_m: float = Field(500.0, gt=0)
    dem_source: str = Field("synthetic", pattern="^(synthetic|real)$")
    dem_path: str | None = None     # path for real data


class LandingAnalyzeRequest(BaseModel):
    slope_redline_deg: float = Field(15.0, gt=0, le=45)
    slope_caution_deg: float = Field(5.0, ge=0, le=30)
    boulder_redline: float = Field(12.0, gt=0)
    ellipse_semi_major_m: float = Field(150.0, gt=0)
    ellipse_semi_minor_m: float = Field(100.0, gt=0)
    top_k_sites: int = Field(5, ge=1, le=20)

    @model_validator(mode="after")
    def validate_thresholds(self):
        if self.slope_caution_deg >= self.slope_redline_deg:
            raise ValueError("slope_caution_deg must be lower than slope_redline_deg")
        if self.ellipse_semi_minor_m > self.ellipse_semi_major_m:
            raise ValueError("ellipse_semi_minor_m cannot exceed ellipse_semi_major_m")
        return self


class RoverPathRequest(BaseModel):
    start_row: int
    start_col: int
    goal_row: int
    goal_col: int
    slope_caution_deg: float = Field(5.0, ge=0, le=30)
    slope_redline_deg: float = Field(15.0, gt=0, le=45)

    @model_validator(mode="after")
    def validate_slopes(self):
        if self.slope_caution_deg >= self.slope_redline_deg:
            raise ValueError("slope_caution_deg must be lower than slope_redline_deg")
        return self


class IceVolumeRequest(BaseModel):
    radar_depth_m:  float = Field(5.0, gt=0, le=50)       # nominal penetration depth
    depth_sigma_m:  float = Field(1.0, ge=0, le=25)       # 1-sigma uncertainty on depth
    conc_min:       float = Field(0.10, ge=0, le=1)       # minimum ice concentration (fraction)
    conc_max:       float = Field(0.20, ge=0, le=1)       # maximum ice concentration (fraction)
    conc_sigma:     float = Field(0.04, ge=0, le=1)       # 1-sigma uncertainty spread
    n_mc:           int   = Field(5000, ge=100, le=100000)

    @model_validator(mode="after")
    def validate_concentration_range(self):
        if self.conc_min > self.conc_max:
            raise ValueError("conc_min cannot exceed conc_max")
        return self


# ── PNG helpers ───────────────────────────────────────────────────────────────

def _array_to_png_bytes(arr: np.ndarray, cmap: str = "gray") -> bytes:
    """Convert a 2-D float/bool array to a PNG byte string."""
    if arr.dtype == bool:
        data = arr.astype(np.uint8) * 255
    else:
        mn, mx = arr.min(), arr.max()
        if mx > mn:
            data = ((arr - mn) / (mx - mn) * 255).astype(np.uint8)
        else:
            data = np.zeros_like(arr, dtype=np.uint8)

    # Apply colormap
    if cmap == "inferno":
        # approximate inferno: black → purple → orange → yellow
        r_lut = np.array([0, 40, 120, 220, 252], dtype=np.uint8)
        g_lut = np.array([0, 11, 28, 110, 255], dtype=np.uint8)
        b_lut = np.array([4, 84, 100, 42, 164], dtype=np.uint8)
        idx = (data / 51).astype(np.int32).clip(0, 4)
        r = r_lut[idx]
        g = g_lut[idx]
        b = b_lut[idx]
        rgba = np.stack([r, g, b, np.full_like(r, 255)], axis=-1)

    elif cmap == "psr":
        # PSR mask: transparent where not PSR, icy-blue where PSR
        alpha = data  # 255 where PSR
        r = np.where(data > 0, 30, 0).astype(np.uint8)
        g = np.where(data > 0, 100, 0).astype(np.uint8)
        b = np.where(data > 0, 200, 0).astype(np.uint8)
        rgba = np.stack([r, g, b, alpha], axis=-1)

    elif cmap == "doubly":
        # Doubly-shadowed: transparent where not DS, bright cyan where DS
        alpha = data
        r = np.where(data > 0, 10, 0).astype(np.uint8)
        g = np.where(data > 0, 220, 0).astype(np.uint8)
        b = np.where(data > 0, 180, 0).astype(np.uint8)
        rgba = np.stack([r, g, b, alpha], axis=-1)

    elif cmap == "lit":
        # Lit fraction: transparent dark → bright gold
        alpha = np.clip(data.astype(int) * 2, 0, 220).astype(np.uint8)
        r = np.clip(data.astype(int) + 80, 0, 255).astype(np.uint8)
        g = np.clip(data.astype(int) - 30, 0, 200).astype(np.uint8)
        b = np.clip(data.astype(int) // 4, 0, 50).astype(np.uint8)
        rgba = np.stack([r, g, b, alpha], axis=-1)

    elif cmap == "redflag":
        # Hard-unsafe mask: transparent where safe, solid danger-red where unsafe
        alpha = data
        r = np.where(data > 0, 239, 0).astype(np.uint8)
        g = np.where(data > 0, 68, 0).astype(np.uint8)
        b = np.where(data > 0, 68, 0).astype(np.uint8)
        rgba = np.stack([r, g, b, alpha], axis=-1)

    elif cmap == "slope":
        # Slope safety: green (flat/safe) -> yellow (caution) -> red (redline+)
        t = data.astype(np.float32) / 255.0
        r = np.clip(255 * np.clip(t * 2.2, 0, 1), 0, 255).astype(np.uint8)
        g = np.clip(255 * (1.0 - np.clip((t - 0.35) * 1.8, 0, 1)), 0, 255).astype(np.uint8)
        b = np.full_like(r, 20)
        rgba = np.stack([r, g, b, np.full_like(r, 200)], axis=-1)

    elif cmap == "boulder":
        # Boulder density: transparent where sparse, hot magenta/orange where dense
        alpha = np.clip(data.astype(int) * 2, 0, 230).astype(np.uint8)
        r = np.clip(60 + data.astype(int), 0, 255).astype(np.uint8)
        g = np.clip(data.astype(int) // 3, 0, 90).astype(np.uint8)
        b = np.clip(120 - data.astype(int) // 2, 0, 150).astype(np.uint8)
        rgba = np.stack([r, g, b, alpha], axis=-1)

    elif cmap == "safety":
        # Composite safety score: red (unsafe) -> yellow -> green (safe)
        t = data.astype(np.float32) / 255.0
        r = np.clip(255 * (1.0 - np.clip((t - 0.4) * 1.8, 0, 1)), 0, 255).astype(np.uint8)
        g = np.clip(255 * np.clip(t * 1.6, 0, 1), 0, 255).astype(np.uint8)
        b = np.full_like(r, 30)
        rgba = np.stack([r, g, b, np.full_like(r, 190)], axis=-1)

    elif cmap == "ohrc":
        # OHRC panchromatic: plain grayscale, fully opaque
        rgba = np.stack([data, data, data, np.full_like(data, 255)], axis=-1)

    elif cmap == "ice_conc":
        # Ice concentration: transparent where zero, cyan→white gradient where ice-rich
        t = data.astype(np.float32) / 255.0
        alpha = np.clip((data.astype(int) * 3), 0, 220).astype(np.uint8)
        r = np.clip((80  + t * 175), 0, 255).astype(np.uint8)
        g = np.clip((200 + t *  55), 0, 255).astype(np.uint8)
        b = np.full_like(r, 240)
        rgba = np.stack([r, g, b, alpha], axis=-1)

    else:  # gray
        rgba = np.stack([data, data, data, np.full_like(data, 255)], axis=-1)

    # Write RGBA PNG using PIL if available, else raw header
    if Image is not None:
        img = Image.fromarray(rgba, mode="RGBA")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    else:
        # Minimal PNG writer
        import zlib, struct

        def write_chunk(chunk_type: bytes, data: bytes) -> bytes:
            length = struct.pack(">I", len(data))
            crc = struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
            return length + chunk_type + data + crc

        h, w = rgba.shape[:2]
        raw_rows = []
        for row in rgba:
            raw_rows.append(b"\x00" + row.tobytes())
        compressed = zlib.compress(b"".join(raw_rows))

        ihdr_data = struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)
        png_bytes = (
            b"\x89PNG\r\n\x1a\n"
            + write_chunk(b"IHDR", ihdr_data)
            + write_chunk(b"IDAT", compressed)
            + write_chunk(b"IEND", b"")
        )
        return png_bytes


# ── Background job ────────────────────────────────────────────────────────────

async def _run_analysis(job_id: str, req: AnalyzeRequest):
    jobs[job_id]["status"] = "running"
    jobs[job_id]["progress"] = 0

    try:
        t0 = time.time()

        # Load DEM
        if req.dem_source == "synthetic":
            dem = make_synthetic_dem(
                grid_size=req.grid_size,
                pixel_size_m=req.pixel_size_m,
            )
        else:
            from real_data_loader import load_dem
            dem = load_dem(req.dem_path)

        jobs[job_id]["progress"] = 10

        # Run detection
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: compute_psr_fast(
                dem,
                pixel_size_m=req.pixel_size_m,
                sun_elevation_deg=req.sun_elevation_deg,
                n_azimuths=req.n_azimuths,
                secondary_illum_threshold=req.secondary_illum_threshold,
                secondary_search_radius_m=req.secondary_search_radius_m,
            )
        )

        jobs[job_id]["progress"] = 80

        # Detect crater candidates
        craters = detect_crater_candidates(
            result["doubly_shadowed_mask"],
            dem,
            req.pixel_size_m,
        )

        jobs[job_id]["progress"] = 90

        total_pixels = int(dem.size)
        psr_pixels = int(result["psr_mask"].sum())
        doubly_shadowed_pixels = int(result["doubly_shadowed_mask"].sum())

        # Cache arrays for PNG rendering
        jobs[job_id]["dem"] = dem
        jobs[job_id]["psr_mask"] = result["psr_mask"]
        jobs[job_id]["doubly_shadowed_mask"] = result["doubly_shadowed_mask"]
        jobs[job_id]["lit_fraction"] = result["lit_fraction"]
        jobs[job_id]["secondary_illum"] = result["secondary_illum"]
        jobs[job_id]["craters"] = craters

        elapsed = time.time() - t0
        jobs[job_id].update({
            "status": "complete",
            "progress": 100,
            "elapsed_s": elapsed,
            "grid_size": req.grid_size,
            "pixel_size_m": req.pixel_size_m,
            "n_craters": len(craters),
            "psr_pixels": psr_pixels,
            "doubly_shadowed_pixels": doubly_shadowed_pixels,
            "psr_area_pct": round(psr_pixels / max(total_pixels, 1) * 100.0, 4),
            "doubly_shadowed_area_pct": round(doubly_shadowed_pixels / max(total_pixels, 1) * 100.0, 4),
        })

    except Exception as e:
        import traceback
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)
        jobs[job_id]["traceback"] = traceback.format_exc()


async def _run_landing_analysis(job_id: str, req: LandingAnalyzeRequest):
    """Step 3: slope + boulder + safety-score + landing-site search, reusing
    the DEM and crater candidates already cached on a completed PSR job."""
    j = jobs[job_id]
    j["landing_status"] = "running"
    j["landing_progress"] = 0

    try:
        t0 = time.time()
        dem = j["dem"]
        pixel_size_m = j["pixel_size_m"]
        craters = j["craters"]

        j["landing_progress"] = 20

        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: run_landing_site_analysis(
                dem,
                pixel_size_m,
                craters,
                slope_redline_deg=req.slope_redline_deg,
                slope_caution_deg=req.slope_caution_deg,
                boulder_redline=req.boulder_redline,
                ellipse_semi_major_m=req.ellipse_semi_major_m,
                ellipse_semi_minor_m=req.ellipse_semi_minor_m,
                top_k_sites=req.top_k_sites,
            )
        )

        j["landing_progress"] = 90

        j["slope_deg"] = result["slope_deg"]
        j["aspect_deg"] = result["aspect_deg"]
        j["slope_redflag"] = result["slope_redflag"]
        j["ohrc_img"] = result["ohrc_img"]
        j["boulder_density"] = result["boulder_density"]
        j["safety_score"] = result["safety_score"]
        j["hard_unsafe"] = result["hard_unsafe"]
        j["landing_sites"] = result["landing_sites"]

        elapsed = time.time() - t0
        j.update({
            "landing_status": "complete",
            "landing_progress": 100,
            "landing_elapsed_s": elapsed,
            "boulder_count": result["boulder_count"],
            "pct_terrain_safe": result["pct_terrain_safe"],
            "pct_slope_redflag": result["pct_slope_redflag"],
            "n_landing_sites": len(result["landing_sites"]),
        })

    except Exception as e:
        import traceback
        j["landing_status"] = "error"
        j["landing_error"] = str(e)
        j["landing_traceback"] = traceback.format_exc()


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/api/analyze")
async def analyze(req: AnalyzeRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "queued",
        "progress": 0,
        "request": req.model_dump(),
    }
    background_tasks.add_task(_run_analysis, job_id, req)
    return {"job_id": job_id, "status": "queued"}


@app.get("/api/status/{job_id}")
async def status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    j = jobs[job_id]
    return {
        "job_id": job_id,
        "status": j["status"],
        "progress": j.get("progress", 0),
        "elapsed_s": j.get("elapsed_s"),
        "n_craters": j.get("n_craters"),
        "error": j.get("error"),
    }


@app.get("/api/result/{job_id}")
async def result(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    j = jobs[job_id]
    if j["status"] != "complete":
        raise HTTPException(409, f"Job status: {j['status']}")
    return {
        "job_id": job_id,
        "status": "complete",
        "elapsed_s": j["elapsed_s"],
        "grid_size": j["grid_size"],
        "pixel_size_m": j["pixel_size_m"],
        "n_craters": j["n_craters"],
        "psr_pixels": j["psr_pixels"],
        "doubly_shadowed_pixels": j["doubly_shadowed_pixels"],
        "psr_area_pct": j["psr_area_pct"],
        "doubly_shadowed_area_pct": j["doubly_shadowed_area_pct"],
        "craters": j["craters"],
    }


@app.get("/api/layer/{job_id}/{layer_name}.png")
async def layer_png(job_id: str, layer_name: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    j = jobs[job_id]
    if j["status"] != "complete":
        raise HTTPException(409, "Not ready")

    layer_map = {
        "dem": (j["dem"], "inferno"),
        "psr": (j["psr_mask"], "psr"),
        "doubly_shadowed": (j["doubly_shadowed_mask"], "doubly"),
        "lit_fraction": (j["lit_fraction"], "lit"),
    }
    if layer_name not in layer_map:
        raise HTTPException(404, f"Unknown layer: {layer_name}")

    arr, cmap = layer_map[layer_name]
    png = _array_to_png_bytes(arr, cmap)
    return Response(content=png, media_type="image/png")


@app.get("/api/dem/{job_id}/profile")
async def dem_profile(job_id: str, row: int | None = None):
    """Return a horizontal elevation profile across the DEM."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    j = jobs[job_id]
    if j["status"] != "complete":
        raise HTTPException(409, "Not ready")
    dem = j["dem"]
    r = row if row is not None else dem.shape[0] // 2
    r = max(0, min(r, dem.shape[0] - 1))
    profile = dem[r, :].tolist()
    return {"row": r, "profile": profile, "pixel_size_m": j["pixel_size_m"]}


# ── Step 3: Landing site analysis ──────────────────────────────────────────

@app.post("/api/landing/{job_id}/analyze")
async def landing_analyze(job_id: str, req: LandingAnalyzeRequest, background_tasks: BackgroundTasks):
    """Kick off slope/boulder/safety-score landing-site analysis for a
    completed PSR job. Reuses that job's DEM and crater candidates."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    j = jobs[job_id]
    if j["status"] != "complete":
        raise HTTPException(409, f"PSR job status: {j['status']} — run /api/analyze first")

    j["landing_status"] = "queued"
    j["landing_progress"] = 0
    background_tasks.add_task(_run_landing_analysis, job_id, req)
    return {"job_id": job_id, "landing_status": "queued"}


@app.get("/api/landing/{job_id}/status")
async def landing_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    j = jobs[job_id]
    return {
        "job_id": job_id,
        "landing_status": j.get("landing_status", "not_started"),
        "landing_progress": j.get("landing_progress", 0),
        "landing_elapsed_s": j.get("landing_elapsed_s"),
        "n_landing_sites": j.get("n_landing_sites"),
        "landing_error": j.get("landing_error"),
    }


@app.get("/api/landing/{job_id}/result")
async def landing_result(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    j = jobs[job_id]
    if j.get("landing_status") != "complete":
        raise HTTPException(409, f"Landing analysis status: {j.get('landing_status', 'not_started')}")
    return {
        "job_id": job_id,
        "landing_status": "complete",
        "landing_elapsed_s": j["landing_elapsed_s"],
        "boulder_count": j["boulder_count"],
        "pct_terrain_safe": j["pct_terrain_safe"],
        "pct_slope_redflag": j["pct_slope_redflag"],
        "landing_sites": j["landing_sites"],
    }


@app.get("/api/landing/{job_id}/layer/{layer_name}.png")
async def landing_layer_png(job_id: str, layer_name: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    j = jobs[job_id]
    if j.get("landing_status") != "complete":
        raise HTTPException(409, "Landing analysis not ready")

    layer_map = {
        "slope": (j["slope_deg"], "slope"),
        "slope_redflag": (j["slope_redflag"], "redflag"),
        "boulder_density": (j["boulder_density"], "boulder"),
        "ohrc": (j["ohrc_img"], "ohrc"),
        "safety_score": (j["safety_score"], "safety"),
        "hard_unsafe": (j["hard_unsafe"], "redflag"),
    }
    if layer_name not in layer_map:
        raise HTTPException(404, f"Unknown layer: {layer_name}")

    arr, cmap = layer_map[layer_name]
    png = _array_to_png_bytes(arr, cmap)
    return Response(content=png, media_type="image/png")


@app.get("/health")
async def health():
    return {"status": "ok", "jobs": len(jobs)}


# ── Step 5: Ice volume estimation ─────────────────────────────────────────────

async def _run_ice_volume(job_id: str, req: IceVolumeRequest):
    """Step 5: Monte-Carlo ice volume estimation from doubly-shadowed pixels."""
    j = jobs[job_id]
    j["ice_status"]   = "running"
    j["ice_progress"] = 10

    try:
        ds_mask      = j.get("doubly_shadowed_mask")
        sec_illum    = j.get("secondary_illum")
        craters      = j.get("craters", [])
        pixel_size_m = j["pixel_size_m"]

        if ds_mask is None or sec_illum is None:
            raise RuntimeError("PSR detection must complete before ice volume estimation.")

        j["ice_progress"] = 30

        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: estimate_ice_volume(
                doubly_shadowed_mask = ds_mask,
                secondary_illum      = sec_illum,
                craters              = craters,
                pixel_size_m         = pixel_size_m,
                radar_depth_m        = req.radar_depth_m,
                depth_sigma_m        = req.depth_sigma_m,
                conc_min             = req.conc_min,
                conc_max             = req.conc_max,
                conc_sigma           = req.conc_sigma,
                n_mc                 = req.n_mc,
            )
        )

        j["ice_progress"] = 90

        # Store concentration map for PNG rendering; strip it from JSON result
        j["ice_conc_map"] = result.pop("conc_map")
        j["ice_result"]   = result
        j.update({"ice_status": "complete", "ice_progress": 100})

    except Exception as e:
        import traceback
        j["ice_status"]     = "error"
        j["ice_error"]      = str(e)
        j["ice_traceback"]  = traceback.format_exc()


@app.post("/api/ice/{job_id}/estimate")
async def ice_estimate(job_id: str, req: IceVolumeRequest, background_tasks: BackgroundTasks):
    """Kick off Step 5 ice-volume estimation for a completed PSR job."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    j = jobs[job_id]
    if j["status"] != "complete":
        raise HTTPException(409, f"PSR job status: {j['status']} — run /api/analyze first")

    j["ice_status"]   = "queued"
    j["ice_progress"] = 0
    j["ice_result"]   = None
    j["ice_conc_map"] = None
    background_tasks.add_task(_run_ice_volume, job_id, req)
    return {"job_id": job_id, "ice_status": "queued"}


@app.get("/api/ice/{job_id}/status")
async def ice_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    j = jobs[job_id]
    return {
        "job_id":       job_id,
        "ice_status":   j.get("ice_status",   "not_started"),
        "ice_progress": j.get("ice_progress", 0),
        "ice_error":    j.get("ice_error"),
    }


@app.get("/api/ice/{job_id}/result")
async def ice_result(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    j = jobs[job_id]
    if j.get("ice_status") != "complete":
        raise HTTPException(409, f"Ice estimation status: {j.get('ice_status', 'not_started')}")
    return {"job_id": job_id, **j["ice_result"]}


@app.get("/api/ice/{job_id}/layer/ice_conc.png")
async def ice_conc_png(job_id: str):
    """Render the per-pixel ice-concentration map as a PNG overlay."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    j = jobs[job_id]
    if j.get("ice_status") != "complete":
        raise HTTPException(409, "Ice estimation not complete")
    png = _array_to_png_bytes(j["ice_conc_map"], cmap="ice_conc")
    return Response(content=png, media_type="image/png")


# ── Step 4: Rover path planning ───────────────────────────────────────────────

async def _run_rover_path(job_id: str, req: RoverPathRequest):
    """Step 4: A* pathfinding from a landing-site pixel to a crater pixel."""
    j = jobs[job_id]
    j["rover_status"] = "running"
    j["rover_progress"] = 10

    try:
        # Requires both PSR job AND landing analysis to be complete
        slope_deg = j.get("slope_deg")
        hard_unsafe = j.get("hard_unsafe")
        pixel_size_m = j["pixel_size_m"]

        if slope_deg is None or hard_unsafe is None:
            raise RuntimeError("Landing analysis must complete before rover path planning.")

        j["rover_progress"] = 30

        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: plan_rover_path(
                slope_deg=slope_deg,
                hard_unsafe=hard_unsafe,
                pixel_size_m=pixel_size_m,
                start_row=req.start_row,
                start_col=req.start_col,
                goal_row=req.goal_row,
                goal_col=req.goal_col,
                slope_caution_deg=req.slope_caution_deg,
                slope_redline_deg=req.slope_redline_deg,
            )
        )

        j["rover_progress"] = 90
        j["rover_result"] = result
        j.update({
            "rover_status": "complete",
            "rover_progress": 100,
        })

    except Exception as e:
        import traceback
        j["rover_status"] = "error"
        j["rover_error"] = str(e)
        j["rover_traceback"] = traceback.format_exc()


@app.post("/api/rover/{job_id}/plan")
async def rover_plan(job_id: str, req: RoverPathRequest, background_tasks: BackgroundTasks):
    """Kick off A* rover-path planning for a job with completed landing analysis."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    j = jobs[job_id]
    if j.get("landing_status") != "complete":
        raise HTTPException(409, "Landing analysis must complete first")

    j["rover_status"] = "queued"
    j["rover_progress"] = 0
    j["rover_result"] = None
    background_tasks.add_task(_run_rover_path, job_id, req)
    return {"job_id": job_id, "rover_status": "queued"}


@app.get("/api/rover/{job_id}/status")
async def rover_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    j = jobs[job_id]
    return {
        "job_id": job_id,
        "rover_status": j.get("rover_status", "not_started"),
        "rover_progress": j.get("rover_progress", 0),
        "rover_error": j.get("rover_error"),
    }


@app.get("/api/rover/{job_id}/result")
async def rover_result(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    j = jobs[job_id]
    if j.get("rover_status") != "complete":
        raise HTTPException(409, f"Rover path status: {j.get('rover_status', 'not_started')}")
    return {"job_id": job_id, **j["rover_result"]}


@app.get("/api/rover/{job_id}/layer/cost_map.png")
async def rover_cost_map_png(job_id: str):
    """Render the traversal-cost map as a PNG (useful for debugging path routing)."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    j = jobs[job_id]
    slope_deg = j.get("slope_deg")
    hard_unsafe = j.get("hard_unsafe")
    if slope_deg is None or hard_unsafe is None:
        raise HTTPException(409, "Landing analysis not complete")

    from rover_path import _build_cost_map
    cost = _build_cost_map(slope_deg, hard_unsafe)
    cost_vis = np.clip(cost, 0, 10) / 10.0  # normalise 0-1 for vis
    # hard-unsafe: cap at 255 as full red
    vis = np.where(hard_unsafe, 255, (cost_vis * 200).astype(np.uint8))
    png = _array_to_png_bytes(vis.astype(np.float32), cmap="gray")
    return Response(content=png, media_type="image/png")
