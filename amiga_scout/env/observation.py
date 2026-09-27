"""What the robot knows when it decides (the policy's inputs): at a block, and at every plant it measures."""
from __future__ import annotations

import math

import numpy as np

from ..actions import SHOOT_ACTIONS
from ..config import FIELD_ACRES

_POSITIONS = [n[len("SHOOT_"):].lower() for n in SHOOT_ACTIONS]


OBS_NAMES = [
    # live camera view at the block's first plant, where the robot stopped (0-1)
    "depth_quality_now", "nir_quality_now",
    # this 10-ft block, as the map sees it before measuring here
    "block_ndvi_vs_threshold", "block_p_stressed", "block_max_p_stressed", "block_uncertain_share",
    "block_ndvi_uncertainty", "block_height_uncertainty",
    # neighbouring blocks (previous block on this pass, next block, the block beside it on the last pass)
    "prev_block_ndvi_vs_threshold", "prev_block_plants", "prev_block_spread", "prev_block_surprise",
    "next_block_p_stressed", "next_block_uncertainty", "other_pass_p_stressed", "other_pass_plants",
    # the map as a whole
    "map_trust", "anomaly_rate", "stressed_share_so_far",
    # batteries and position
    "battery_left", "spare_sets_left", "spare_plants_per_block", "protocol_energy_ratio", "time_to_next_zap",
    "blocks_done_frac", "pass_progress", "position_in_pass",
    # field, known in advance from the GPS map of the rows
    "field_acres",
    # the decision being made: at a block (0) or at a plant (1); the current plant's situation
    "at_plant", "careful_block", "plant_readings", "plants_left_in_plan", "camera_offset_now",
    # live quality readings of this plant from the positions it may shoot from (-1 = not seen yet)
    *[f"depth_q_{p}" for p in _POSITIONS], *[f"nir_q_{p}" for p in _POSITIONS],
    # its last shot, and what the map still does not know about it
    "last_depth_quality", "last_nir_quality", "plant_ndvi_uncertainty", "plant_p_stressed", "plant_height_uncertainty",
]


class ObservationMixin:
    """The observation vector (used by AmigaMissionEnv)."""

    def _obs(self):
        """What the robot knows when it decides (OBS_NAMES, same order)."""
        c = self.cfg
        K, thr = self.K, c.ndvi_stress_threshold
        b = self.blk
        bp = slice(b * K, b * K + K)
        ps = self.pstress[bp]
        sdh0 = math.sqrt(c.map_height_spatial_sd_cm ** 2 + c.map_height_plant_sd_cm ** 2)
        block = ((self.blk_mu_n[b] / K - thr) / 0.2, float(ps.mean()), float(ps.max()),
                 float(((ps > 0.05) & (ps < 0.95)).mean()), self.blk_sd_n[b] / K / self.sd0,
                 float(self.sd_h[bp].mean()) / sdh0)
        prev_b, next_b = self._block_at(self.pos - K), self._block_at(self.pos + K)
        if prev_b is not None:
            reads = self.block_readings(prev_b)
            prev = ((self.blk_mu_n[prev_b] / K - thr) / 0.2, self.blk_count[prev_b] / K,
                    float(reads.std()) / 0.1 if reads.size >= 2 else 0.0, min(self.blk_maxz[prev_b], 5.0) / 5.0)
        else:
            prev = (0.0, 0.0, 0.0, 0.0)
        nxt = ((float(self.pstress[next_b * K:next_b * K + K].mean()), self.blk_sd_n[next_b] / K / self.sd0)
               if next_b is not None else (0.0, 0.0))
        ob = (self.row - 1) * self.B + self._col(self.row, self.pos) // K
        if self.row > 0 and self.vblock[ob]:
            other = (float(self.pstress[ob * K:ob * K + K].mean()), self.blk_count[ob] / K)
        else:
            other = (0.0, 0.0)
        meas = self.prec_n > 0
        stressed_share = float((self.wsum_n[meas] / self.prec_n[meas] < thr).mean()) if meas.any() else 0.0
        need = self.protocol_plan()[1]
        avail = self.energy_available()
        return np.array([
            self.pv_qd, self.pv_qn, *block, *prev, *nxt, *other,
            math.log(self.trust) / math.log(c.trust_max), self.anomalies / max(self.n_scanned, 1), stressed_share,
            self.batt_wh / self.capacity_wh, self.sets_left / max(c.spare_pack_sets, 1),
            float(np.clip(self.spare_plants_per_block() / (K - 1), -1.0, 2.0)), min(need / max(avail, 1e-6), 3.0) / 3.0,
            (1.0 - self.zap_timer / self.zap_every_s) if self.zap_s > 0 else 1.0,
            (int(self.route_rank[b]) + 1) / self.NBv, self.row / max(self.R - 1, 1),
            (self.pos - int(self.p0[self.row])) / max(int(self.p1[self.row] - self.p0[self.row]), 1), c.field_acres / FIELD_ACRES,
            *self._plant_obs(sdh0),
        ], dtype=np.float32).clip(-5.0, 5.0)    # keep the declared range: a very noisy reading can exceed it

    def _plant_obs(self, sdh0):
        at_plant = self.phase == "plant"
        i = self.here
        left = (len(self.plan) - self.plan_k - 1) / self.K if at_plant else 0.0
        return (float(at_plant), float(at_plant and self.block_mode == "careful"), min(self.plant_shots, 2) / 2.0,
                left, (self.x - self.x0) / 0.3,
                *np.nan_to_num(self.seen_qd, nan=-1.0), *np.nan_to_num(self.seen_qn, nan=-1.0),
                self.last_qd, self.last_qn, self.sd_n[i] / self.sd0, float(self.pstress[i]), self.sd_h[i] / sdh0)
