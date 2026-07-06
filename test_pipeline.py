import unittest

import numpy as np

from horizon_shadow import compute_psr_fast, detect_crater_candidates, make_synthetic_dem
from ice_volume import estimate_ice_volume
from landing_site import run_landing_site_analysis
from rover_path import plan_rover_path


class PipelineTests(unittest.TestCase):
    def test_synthetic_pipeline_end_to_end(self):
        pixel_size_m = 10.0
        dem = make_synthetic_dem(grid_size=128, pixel_size_m=pixel_size_m)

        shadow = compute_psr_fast(
            dem,
            pixel_size_m=pixel_size_m,
            sun_elevation_deg=1.5,
            n_azimuths=8,
            secondary_illum_threshold=0.05,
            secondary_search_radius_m=400.0,
        )

        self.assertGreater(int(shadow["psr_mask"].sum()), 0)
        self.assertGreater(int(shadow["doubly_shadowed_mask"].sum()), 0)
        self.assertEqual(shadow["lit_fraction"].shape, dem.shape)
        self.assertEqual(shadow["secondary_illum"].shape, dem.shape)

        craters = detect_crater_candidates(
            shadow["doubly_shadowed_mask"],
            dem,
            pixel_size_m,
        )
        self.assertGreaterEqual(len(craters), 1)
        self.assertGreater(craters[0]["confidence"], 0.5)

        landing = run_landing_site_analysis(
            dem,
            pixel_size_m,
            craters,
            slope_redline_deg=15.0,
            slope_caution_deg=5.0,
            boulder_redline=12.0,
            ellipse_semi_major_m=150.0,
            ellipse_semi_minor_m=100.0,
            top_k_sites=5,
        )
        self.assertGreaterEqual(len(landing["landing_sites"]), 1)
        self.assertEqual(landing["slope_deg"].shape, dem.shape)
        self.assertEqual(landing["hard_unsafe"].shape, dem.shape)

        start = landing["landing_sites"][0]
        target = craters[0]
        traverse = plan_rover_path(
            landing["slope_deg"],
            landing["hard_unsafe"],
            pixel_size_m,
            start_row=start["centroid_row"],
            start_col=start["centroid_col"],
            goal_row=round(target["centroid_row"]),
            goal_col=round(target["centroid_col"]),
            slope_caution_deg=5.0,
            slope_redline_deg=15.0,
        )
        self.assertTrue(traverse["path_found"], traverse.get("diagnosis"))
        self.assertGreater(traverse["total_distance_m"], 0.0)
        self.assertGreaterEqual(len(traverse["waypoints"]), 2)

        ice = estimate_ice_volume(
            shadow["doubly_shadowed_mask"],
            shadow["secondary_illum"],
            craters,
            pixel_size_m,
            radar_depth_m=5.0,
            depth_sigma_m=1.0,
            conc_min=0.10,
            conc_max=0.20,
            conc_sigma=0.04,
            n_mc=500,
            seed=7,
        )
        self.assertEqual(ice["n_ice_pixels"], int(shadow["doubly_shadowed_mask"].sum()))
        self.assertGreater(ice["total_vol_m3_p50"], 0.0)
        self.assertLessEqual(ice["total_vol_m3_p5"], ice["total_vol_m3_p50"])
        self.assertLessEqual(ice["total_vol_m3_p50"], ice["total_vol_m3_p95"])
        self.assertEqual(ice["conc_map"].shape, dem.shape)
        self.assertGreaterEqual(len(ice["per_crater"]), 1)

        per_region_area = sum(row["area_m2"] for row in ice["per_crater"])
        self.assertLessEqual(per_region_area, ice["total_area_m2"] + 1e-6)

    def test_ice_volume_zero_mask_and_validation(self):
        mask = np.zeros((16, 16), dtype=bool)
        illum = np.ones((16, 16), dtype=np.float32)

        ice = estimate_ice_volume(mask, illum, [], pixel_size_m=10.0, n_mc=200)
        self.assertEqual(ice["n_ice_pixels"], 0)
        self.assertEqual(ice["total_vol_m3_p50"], 0.0)
        self.assertEqual(len(ice["per_crater"]), 0)

        with self.assertRaises(ValueError):
            estimate_ice_volume(mask, illum, [], pixel_size_m=10.0, conc_min=0.3, conc_max=0.2)

    def test_rover_reports_unreachable_out_of_bounds(self):
        slope = np.zeros((12, 12), dtype=np.float32)
        unsafe = np.zeros((12, 12), dtype=bool)

        result = plan_rover_path(
            slope,
            unsafe,
            pixel_size_m=10.0,
            start_row=-1,
            start_col=0,
            goal_row=5,
            goal_col=5,
        )
        self.assertFalse(result["path_found"])
        self.assertIn("outside", result["diagnosis"]["reason"])


if __name__ == "__main__":
    unittest.main()
