"""
End-to-end HTTP integration test against a *running* FastAPI server.

Unlike test_pipeline.py (which calls the pure-Python functions directly and
needs no server), this test exercises the actual HTTP surface: request
validation, status polling, PNG layer rendering, and the full five-step
pipeline (detect -> landing -> rover -> ice volume) wired through real
network calls.

This is opt-in: it's skipped automatically if no server is reachable at
TEST_SERVER_URL, so `python -m unittest discover -s tests` stays fast and
dependency-light by default.

To run it:
    uvicorn main:app --host 127.0.0.1 --port 8000 &
    python -m unittest tests.test_http_integration -v
"""

import os
import time
import unittest
import urllib.error
import urllib.request
import json

BASE = os.environ.get("TEST_SERVER_URL", "http://127.0.0.1:8000")


def _server_reachable() -> bool:
    try:
        urllib.request.urlopen(f"{BASE}/health", timeout=1.0)
        return True
    except Exception:
        return False


def _post(path: str, payload: dict) -> tuple[int, dict]:
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _get(path: str) -> tuple[int, dict]:
    try:
        with urllib.request.urlopen(f"{BASE}{path}", timeout=10) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _poll(path: str, status_key: str, timeout_s: float = 30.0) -> dict:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        _, body = _get(path)
        if body.get(status_key) in ("complete", "error"):
            return body
        time.sleep(0.25)
    raise TimeoutError(f"Polling {path} timed out")


@unittest.skipUnless(_server_reachable(), f"No server reachable at {BASE} — skipping HTTP integration test.")
class HttpIntegrationTests(unittest.TestCase):
    def test_full_pipeline_over_http(self):
        # Step 1/2: detection
        status, body = _post("/api/analyze", {
            "grid_size": 128, "pixel_size_m": 10.0, "sun_elevation_deg": 1.5,
            "n_azimuths": 8, "secondary_illum_threshold": 0.05,
            "secondary_search_radius_m": 300.0, "dem_source": "synthetic",
        })
        self.assertEqual(status, 200)
        job_id = body["job_id"]

        s = _poll(f"/api/status/{job_id}", "status")
        self.assertEqual(s["status"], "complete", s.get("error"))

        rstatus, result = _get(f"/api/result/{job_id}")
        self.assertEqual(rstatus, 200)
        self.assertGreaterEqual(len(result["craters"]), 1)
        self.assertIn("psr_area_pct", result)
        self.assertIn("doubly_shadowed_area_pct", result)

        for layer in ("dem", "psr", "doubly_shadowed", "lit_fraction"):
            lstatus, _ = _get_raw(f"/api/layer/{job_id}/{layer}.png")
            self.assertEqual(lstatus, 200)

        # Step 3: landing sites
        status, _ = _post(f"/api/landing/{job_id}/analyze", {
            "slope_redline_deg": 15.0, "slope_caution_deg": 5.0, "boulder_redline": 12.0,
            "ellipse_semi_major_m": 150.0, "ellipse_semi_minor_m": 100.0, "top_k_sites": 5,
        })
        self.assertEqual(status, 200)
        ls = _poll(f"/api/landing/{job_id}/status", "landing_status")
        self.assertEqual(ls["landing_status"], "complete", ls.get("landing_error"))

        _, landing_result = _get(f"/api/landing/{job_id}/result")
        self.assertGreaterEqual(len(landing_result["landing_sites"]), 1)

        # Step 4: rover path
        start_site = landing_result["landing_sites"][0]
        target = result["craters"][0]
        status, _ = _post(f"/api/rover/{job_id}/plan", {
            "start_row": start_site["centroid_row"], "start_col": start_site["centroid_col"],
            "goal_row": round(target["centroid_row"]), "goal_col": round(target["centroid_col"]),
            "slope_caution_deg": 5.0, "slope_redline_deg": 15.0,
        })
        self.assertEqual(status, 200)
        rs = _poll(f"/api/rover/{job_id}/status", "rover_status")
        self.assertEqual(rs["rover_status"], "complete", rs.get("rover_error"))

        _, rover_result = _get(f"/api/rover/{job_id}/result")
        self.assertIsInstance(rover_result["path_found"], bool)

        # Step 5: ice volume
        status, _ = _post(f"/api/ice/{job_id}/estimate", {
            "radar_depth_m": 5.0, "depth_sigma_m": 1.0,
            "conc_min": 0.10, "conc_max": 0.20, "conc_sigma": 0.04, "n_mc": 1000,
        })
        self.assertEqual(status, 200)
        ics = _poll(f"/api/ice/{job_id}/status", "ice_status")
        self.assertEqual(ics["ice_status"], "complete", ics.get("ice_error"))

        _, ice_result = _get(f"/api/ice/{job_id}/result")
        self.assertLessEqual(ice_result["total_vol_m3_p5"], ice_result["total_vol_m3_p50"])
        self.assertLessEqual(ice_result["total_vol_m3_p50"], ice_result["total_vol_m3_p95"])
        self.assertGreaterEqual(len(ice_result["per_crater"]), 1)

    def test_validation_errors_return_422_with_detail(self):
        status, body = _post("/api/analyze", {"grid_size": 10})  # below minimum
        self.assertEqual(status, 422)
        self.assertIn("detail", body)

        status, body = _post("/api/analyze", {})  # placeholder job for landing validation test
        job_id = body["job_id"]
        _poll(f"/api/status/{job_id}", "status")

        status, body = _post(f"/api/landing/{job_id}/analyze", {
            "slope_caution_deg": 20.0, "slope_redline_deg": 15.0,
        })
        self.assertEqual(status, 422)

        status, body = _post(f"/api/ice/{job_id}/estimate", {
            "conc_min": 0.5, "conc_max": 0.2,
        })
        self.assertEqual(status, 422)

    def test_unknown_job_returns_404(self):
        status, _ = _get("/api/status/does-not-exist")
        self.assertEqual(status, 404)


def _get_raw(path: str) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(f"{BASE}{path}", timeout=10) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


if __name__ == "__main__":
    unittest.main()
