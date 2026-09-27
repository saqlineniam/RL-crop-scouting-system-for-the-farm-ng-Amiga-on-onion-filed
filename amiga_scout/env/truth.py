"""The hidden truth of a mission (what the robot does NOT know): every plant's NDVI and height, from the
real drone map or a synthetic trend, plus the field type's added stress; and the day's robot/weather conditions."""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from ..fields import field_map


class TruthMixin:
    """The hidden field truth and the day's conditions (used by AmigaMissionEnv)."""

    # ---------------------------------------------------------------- truth
    def _random_conditions(self, c):
        """A different robot/weather day (training with --randomize); ranges stop short of PERTURBATIONS."""
        f = lambda lo, hi: float(self.np_random.uniform(lo, hi))  # noqa: E731
        speed = f(0.85, 1.1)
        return replace(c, row_speed_mps=c.row_speed_mps * speed, travel_speed_mps=c.travel_speed_mps * speed,
                       hop_overhead_s=c.hop_overhead_s * f(0.8, 1.4), drive_power_w=c.drive_power_w * f(0.9, 1.35),
                       usable_capacity_frac=min(1.0, c.usable_capacity_frac * f(0.85, 1.05)),
                       wind=c.wind * f(0.6, 1.8), depth_subpixel_rms=c.depth_subpixel_rms * f(0.8, 1.3),
                       glare_max_share=c.glare_max_share * f(0.5, 1.5),
                       arrival_jitter_m=c.arrival_jitter_m * f(0.8, 2.0), plant_spot_sd_m=c.plant_spot_sd_m * f(0.8, 1.3),
                       zap_duration_min=c.zap_duration_min * f(0.9, 1.25))

    def _make_truth(self):
        """The hidden truth at every plant of the route. Real field: the drone map carried over to the robot's
        scale (relative differences only) + each plant's own deviation; synthetic field: a smooth random trend
        instead. Then the field type's stress (patches / spots) is added - with exact ground truth."""
        c, rng = self.cfg, self.np_random
        rr, cc = np.divmod(np.arange(self.N), self.C)
        x, y = (cc + 0.5) * self.dxp, (rr + 0.5) * self.dy
        if self.real:
            f = field_map(c.field)
            gap = self.gap
            base_n = np.where(gap, -0.30, c.drone_ndvi_gain * (np.nan_to_num(self.bg_leaf, nan=f["leaf_median"]) - f["leaf_median"]))
            size = np.clip(np.maximum(self.bg_cover, 1e-3) / f["cover_median"], 0.0, None) ** c.height_cover_exponent
            size = np.where(gap, 0.4, np.clip(size, 0.4, 1.6))
            trend = np.where(self.vplant, base_n, 0.0)          # the real field's own pattern, around its median
            h_scale = np.where(self.vplant, size, 1.0)
        else:
            trend = np.zeros(self.N)
            for _ in range(6):
                k = 2 * np.pi / (c.trend_wavelength_m * rng.uniform(0.6, 1.6))
                ang = rng.uniform(0, 2 * np.pi)
                trend += np.cos(k * (np.cos(ang) * x + np.sin(ang) * y) + rng.uniform(0, 2 * np.pi))
            trend *= c.trend_ndvi_sd / np.sqrt(3.0)
            h_scale = np.ones(self.N)
        stress = np.zeros(self.N)
        for _ in range(rng.poisson(c.patches_per_acre * c.field_acres)):
            cx, cy = rng.uniform(0, self.C * self.dxp), rng.uniform(0, self.width_m)
            rad = rng.uniform(c.patch_radius_m_min, c.patch_radius_m_max)
            drop = rng.uniform(c.patch_ndvi_drop_min, c.patch_ndvi_drop_max)
            stress = np.maximum(stress, drop * np.exp(-0.5 * ((x - cx) ** 2 + (y - cy) ** 2) / rad ** 2))
        if c.spot_stress_prob > 0:
            spot = rng.random(self.N) < c.spot_stress_prob
            stress = np.where(spot, np.maximum(stress, rng.uniform(0.2, 0.4, self.N)), stress)
        self.ndvi = np.clip(c.healthy_ndvi + trend + rng.normal(0, c.ndvi_plant_sd, self.N) - stress, 0.05, 1.0)
        self.height = np.clip((c.height_mean_cm * h_scale + rng.normal(0, c.height_plant_sd_cm, self.N))
                              * (1.0 - c.stress_height_drop_frac * np.clip(stress / 0.45, 0, 1)), 5.0, None)
        self.stressed = (self.ndvi < c.ndvi_stress_threshold) & self.vplant
        # what the cameras will meet at every plant (unknown to the robot; see amiga_scout/sensors/shot.py):
        self.plant_x = rng.normal(0.0, c.plant_spot_sd_m, self.N)          # plant centre vs its nominal spot (m)
        self.plant_y = rng.uniform(0.0, c.row_spacing_across_m / 2, self.N)  # its row, across from the cameras (m)
        self.leaf_w = c.leaf_width_mm / 1e3 * np.clip(1.0 + c.leaf_width_cv * rng.standard_normal(self.N), 0.4, 1.8)
        self.glare_amp = rng.uniform(0.0, 1.0, self.N)                     # how glossy / how turned to the lamp
        self.glare_x = self.plant_x + c.lamp_offset_m / 2 + rng.normal(0.0, c.glare_width_m, self.N)
        self.view_g = rng.standard_normal((self.N, 4))                     # viewpoint errors: height (2), NDVI (2)
        self.calibration = float(rng.normal(0.0, c.ndvi_calibration_sd))  # this mission's NDVI calibration offset
        self.n_stressed = int(self.stressed.sum())
        # the lab's sample values: true mean of each 10-ft block; and each part's truth (zone Z = outside the crop)
        self.blk_true_n = self.ndvi.reshape(self.NB, self.K).mean(1)
        self.blk_true_h = self.height.reshape(self.NB, self.K).mean(1)
        self.z_true_mean = np.bincount(self.zone_of, weights=self.ndvi, minlength=self.Z + 1) / self.zone_size
        frac = np.bincount(self.zone_of, weights=self.stressed.astype(float), minlength=self.Z + 1) / self.zone_size
        self.z_flag_true = frac >= c.part_stress_frac
        self.z_flag_true[self.Z] = False
