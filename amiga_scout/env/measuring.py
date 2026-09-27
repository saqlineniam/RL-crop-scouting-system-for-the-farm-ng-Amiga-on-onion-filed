"""Measuring a plant: what the cameras see from each position (the camera model is in amiga_scout/sensors), the live
quality readings, the robot's own positioning (a little back or forward before each shot - the RL's choice), and
shots (YOLO + D455 depth -> height, red + NIR -> NDVI) that update the robot's map.

One plant, step by step:
    arrive    the robot stops near the plant. While driving up it streamed the cameras, so it already has (noisy)
              quality readings for the positions it passed over (SHOOT_OFFSETS_M <= 0); positions ahead are unknown
    SHOOT_x   move to offset x (0.1-m moves, reverse allowed) and shoot: one frame (quick block) or frames averaged
              (careful block). Positions passed on the way get read too
    then      quick block: on to the next plant. Careful block: a second SHOOT from another position, or DONE
    measured  a plant counts as measured (the protocol's sample) once it has an NDVI reading; height is added when
              the D455 can give it (not closer than its minimum distance, plant in its view)
    safety    if two shots gave no NDVI (the plant outside the NIR camera's view), the next attempt is centred on
              the plant's YOLO box - so every planned plant is measured (the protocol never fails)
"""
from __future__ import annotations

import math

import numpy as np

from ..actions import shoot_offset
from ..config import SHOOT_OFFSETS_M
from ..sensors import depth_quality, height_shot, ndvi_shot, nir_quality, p_detect, view_angle, viewpoint_error

OFFSETS = np.array(SHOOT_OFFSETS_M)


class MeasureMixin:
    """Stopping, positioning and shooting at a plant (used by AmigaMissionEnv)."""

    # ---------------------------------------------------------------- what the cameras see
    def _view(self, i, x=None):
        """Plant i seen from camera position x (m along the row from the plant's nominal spot):
        (dx along the row from the plant's centre, y across the bed, distance to its top, distance to mid-canopy)."""
        c = self.cfg
        x = self.x if x is None else x
        h = self.height[i] / 100.0
        return (x - self.plant_x[i], self.plant_y[i], max(c.camera_height_m - h, 0.02),
                max(c.camera_height_m - h / 2, 0.02))

    def _quality(self, i, x=None):
        """(depth quality, NIR quality, mixed share, glare share) of plant i from camera position x. The two
        qualities are what the real robot computes live: the depth fill rate on the plant (0 = closer than the
        D455's minimum distance, or outside its view) and the share of clean leaf pixels in the NIR image."""
        c = self.cfg
        dx, y, z_top, z_mid = self._view(i, x)
        qn, mixed, glare = nir_quality(c, z_mid, dx, y, self.leaf_w[i], self.glare_amp[i],
                                       self.glare_x[i] - self.plant_x[i])
        return depth_quality(c, z_top, dx, y, self.leaf_w[i]), qn, mixed, glare

    # ---------------------------------------------------------------- measurement noise
    def _rng(self, *key):
        """The random numbers of one reading, fixed by WHAT is read: (kind, plant, shot number or position).
        So a copy of the mission (fork) that measures the same plant gets the same noise - common random numbers,
        which lets training compare two choices from the same moment without the measurement noise drowning the
        difference (PEGASUS, Ng & Jordan 2000; the 'vine' of TRPO, Schulman et al. 2015). noise_salt gives a copy
        a fresh, independent set of readings for everything still to come."""
        return np.random.default_rng([self.noise_seed, self.noise_salt, *key])

    def _read(self, x, noise):
        """Live (depth, NIR) quality readings of the current plant from position x, with reading noise."""
        qd, qn, _, _ = self._quality(self.here, x)
        e = self._rng(1, self.here, int(round(x * 1000)) + 100000, int(noise * 1e4)).normal(0, noise, 2)
        return float(np.clip(qd + e[0], 0, 1)), float(np.clip(qn + e[1], 0, 1))

    def _update_preview(self):
        """The live readings where the robot stands now (also the block decision's first view)."""
        self.pv_qd, self.pv_qn = self._read(self.x, self.cfg.preview_noise)
        k = self._offset_index(self.x - self.x0)
        if k is not None:
            self.seen_qd[k], self.seen_qn[k] = self.pv_qd, self.pv_qn

    def _offset_index(self, offset):
        k = int(np.argmin(np.abs(OFFSETS - offset)))
        return k if abs(OFFSETS[k] - offset) < 1e-6 else None

    # ---------------------------------------------------------------- arriving at a plant
    def _arrive(self, resume=False):
        """Stopped at a plant: somewhere near (usually a little past) its nominal spot. The positions it passed over
        while driving up were read from the moving cameras (noisier); positions ahead are not known yet. `resume`:
        back at the same plant after a battery swap - what was already measured there is kept."""
        c = self.cfg
        jitter = self._rng(2, self.here, int(resume), int(self.shots[self.here])).normal(0, c.arrival_jitter_m)
        self.x = self.x0 = c.arrival_offset_m + float(jitter)
        self.seen_qd, self.seen_qn = np.full(len(OFFSETS), np.nan), np.full(len(OFFSETS), np.nan)
        for k, o in enumerate(OFFSETS):
            if o < 0:
                self.seen_qd[k], self.seen_qn[k] = self._read(self.x0 + o, c.preview_noise_moving)
        if not resume:
            self.last_qd = self.last_qn = 0.0
            self.last_detected = False
            self.plant_shots = self.plant_moves = self.plant_attempts = 0
            self.plant_read = False                      # has this plant given a reading yet?
        self._update_preview()

    def _move_to(self, x):
        """Move along the row to camera position x in 0.1-m moves (forward or in reverse), reading the positions
        passed on the way."""
        c = self.cfg
        n = int(round(abs(x - self.x) / c.nudge_step_m))
        for _ in range(n):
            self.x += math.copysign(c.nudge_step_m, x - self.x)
            self.nudges_total += 1
            self.plant_moves += 1
            self._work(self.t_nudge, True)
            if self.ended:
                return
            self._update_preview()
        self.x = x

    def move_cost_wh(self, offset):
        """Energy (Wh) of moving from where the cameras are now to `offset` from where the robot stopped."""
        n = int(round(abs(self.x0 + offset - self.x) / self.cfg.nudge_step_m))
        return self._est(n * self.t_nudge, 0.0)[1]

    # ---------------------------------------------------------------- one decision at a plant
    def _plant_shot(self, action):
        """SHOOT_x: move to that position and shoot. If the packs reached the return level, the lab's battery
        procedure comes first (drive home, swap, come back to this plant)."""
        if self.ended or (self.batt_wh <= self.reserve_wh and not self._battery_trip()):
            return
        i = self.here
        target = self.x0 + shoot_offset(action)
        if self.plant_attempts >= 2 and not self.plant_read:
            target = self.plant_x[i]                     # safety: centre on the plant's YOLO box (NIR sees it there)
        self._move_to(target)
        if self.ended:
            return
        self._shoot(i, self.block_mode == "careful")
        self.plant_attempts += 1
        self.shift_sum += abs(self.x - self.x0)
        self.shift_n += 1

    def _end_plant(self):
        """Bookkeeping when the robot leaves a plant."""
        if self.plant_read:
            careful = self.block_mode == "careful"
            mode = self.block_mode                       # the robot's own running averages, used for planning
            self.avg_moves[mode] += 0.05 * (self.plant_moves - self.avg_moves[mode])
            self.avg_shots[mode] += 0.05 * (self.plant_attempts - self.avg_shots[mode])
            self.avg_to_read["moves"] += 0.05 * (self.read_moves - self.avg_to_read["moves"])
            self.avg_to_read["shots"] += 0.05 * (self.read_attempts - self.avg_to_read["shots"])
            self.careful_plants += careful
            self.quick_plants += not careful
            self.second_shots += self.plant_shots >= 2

    def plant_done(self):
        """Is the robot finished with this plant without a further decision? A quick plant after its reading, a
        careful plant after two shots."""
        if not self.plant_read:
            return self.plant_attempts >= 6              # (only if YOLO can miss) give up; the block rule takes over
        return self.block_mode == "quick" or self.plant_shots >= 2

    # ---------------------------------------------------------------- a shot
    def _shoot(self, i, careful=False):
        """One shot from the current camera position: YOLO finds the plant (always, unless YOLO_CAN_MISS), the D455
        depth gives height (unless the plant is too close or out of its view) and the red + NIR images give NDVI
        (unless out of view) - each with the errors the camera model gives for this position
        (amiga_scout/sensors/shot.py). A careful shot averages frames for careful_average_s."""
        c = self.cfg
        self._work(c.shot_s + (c.careful_average_s if careful else 0.0), False)
        if self.ended:
            return
        dx, y, z_top, z_mid = self._view(i)
        qd, qn, mixed, glare = self._quality(i)
        self.shots[i] += 1
        rng = self._rng(3, i, int(self.shots[i]))          # this plant's n-th shot: the same noise in any copy
        self.last_qd, self.last_qn = qd, qn
        self.last_detected = bool(rng.random() < p_detect(c, z_top))
        if not self.last_detected:
            self.failed_shots += 1
            return
        g = self.view_g[i]
        yh, sh = height_shot(c, rng, self.height[i], z_top, view_angle(dx, y, z_top), qd, self.leaf_w[i],
                             c.leaf_width_mm / 1e3, viewpoint_error(g[0], g[1], dx, c.view_corr_m), careful)
        yn, sn = ndvi_shot(c, rng, self.ndvi[i], qn, mixed, glare, z_mid, self.leaf_w[i],
                           viewpoint_error(g[2], g[3], dx, c.view_corr_m), self.calibration, careful)
        if yh is None and yn is None:                    # the plant was out of view: no reading from this shot
            return
        # TRUST: if readings keep landing further from the map's prediction than their own uncertainty says
        # they should, the field is not behaving like the model -> widen the map's uncertainty.
        z2n = (yn - self.mu_n[i]) ** 2 / (self.sd_n[i] ** 2 + sn ** 2) if yn is not None else None
        z2h = (yh - self.mu_h[i]) ** 2 / (self.sd_h[i] ** 2 + sh ** 2) if yh is not None else None
        z2 = [v for v in (z2n, z2h) if v is not None]
        self.surprise += c.trust_adapt_rate * (min(sum(z2) / len(z2), 25.0) - self.surprise)
        self.trust = float(np.clip(self.surprise, 1.0, c.trust_max))
        # ANOMALY: a reading far from its own prediction loosens THAT plant from its neighbours, so its
        # own measurements decide its value and it does not drag the neighbours with it.
        mult = min(max(z2) / c.anomaly_z2, c.anomaly_max)
        if mult > self.nug[i]:
            self.anomalies += int(self.nug[i] == 1.0 and mult > 1.0)
            self.nug[i] = mult
        b = int(self.block_of[i])
        if yn is not None:                               # an NDVI reading: the plant is measured (protocol sample)
            self.blk_maxz[b] = max(self.blk_maxz[b], math.sqrt(z2n))
            if self.scan_count[i] == 0:
                self.n_scanned += 1
                self.blk_count[b] += 1
            self.scan_count[i] += 1
            self.plant_shots += 1
            if not self.plant_read:                      # what it took to get this plant's first reading
                self.read_attempts, self.read_moves = self.plant_attempts + 1, self.plant_moves
            self.plant_read = True
        self.best_qd[i], self.best_qn[i] = max(self.best_qd[i], qd), max(self.best_qn[i], qn)
        # all shots of a plant are fused, each weighted by the error the robot attributes to it
        if yn is not None:
            self.prec_n[i] += 1.0 / sn ** 2
            self.wsum_n[i] += yn / sn ** 2
        if yh is not None:
            self.prec_h[i] += 1.0 / sh ** 2
            self.wsum_h[i] += yh / sh ** 2
        self._refresh_map(self.row, self._col(self.row, self.pos))
