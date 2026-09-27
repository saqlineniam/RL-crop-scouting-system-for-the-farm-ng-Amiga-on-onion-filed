"""AmigaMissionEnv: one scouting mission as a Gymnasium environment - one step = one decision (a block's plan, or
where to shoot a plant from).
The parts live in the mixins next to this file (route, energy, truth, measuring, mapping, observation)."""
from __future__ import annotations

import copy
from dataclasses import replace
import math

from gymnasium import spaces
import gymnasium as gym
import numpy as np

from ..actions import A, ACTIONS, ACTION_BRANCH, DONE, SHOOT_HERE, decode, encode, is_block_action
from ..config import Cfg, SCENARIOS
from ..fields import field_pool
from ..mathutil import _f1, _phi
from .energy import EnergyMixin
from .mapping import MapMixin
from .measuring import MeasureMixin
from .observation import OBS_NAMES, ObservationMixin
from .route import RouteMixin
from .truth import TruthMixin


class AmigaMissionEnv(RouteMixin, EnergyMixin, TruthMixin, MeasureMixin, MapMixin, ObservationMixin, gym.Env):
    """One scouting mission (see the package docstring). One step = one decision: at a block, how carefully and
    how many plants; at a plant, where to shoot from (or DONE). The reward is the change of ACCURACY minus the
    time it cost.

    positioner: if set (a function env -> plant action), the plant decisions are made by it inside step(), so the
    caller only makes the block decisions (the rules' positioning, or a learned positioning policy)."""
    metadata = {"render_modes": []}
    # what never changes during a mission: shared, not copied, by fork()
    _FIXED = ("positioner", "cfg", "base_cfg", "pool", "scenario_mix", "bg_leaf", "bg_cover", "gap", "vplant", "vblock",
              "block_of", "zone_of", "zone_size", "route_rank", "p0", "p1", "c0", "c1", "kblk", "krow", "map_info",
              "ndvi", "height", "stressed", "plant_x", "plant_y", "leaf_w", "glare_amp", "glare_x", "view_g",
              "blk_true_n", "blk_true_h", "z_true_mean", "z_flag_true", "observation_space", "action_space")

    def __init__(self, cfg: Cfg | None = None, scenario_mix: list | None = None, randomize: bool = False,
                 positioner=None):
        super().__init__()
        self.cfg = cfg or Cfg()
        # cfg.field: the field, or a pool of fields to draw from every episode (real maps and/or squares);
        # scenario_mix: a different field type every episode (like the paper's "global policy");
        # randomize: different robot/weather conditions every episode (sim-to-real robustness).
        self.base_cfg, self.scenario_mix, self.randomize = self.cfg, scenario_mix, randomize
        self.positioner = positioner
        self.pool = field_pool(self.cfg.field)
        self._geom_key = None
        self.actions, self.action_branch, self.obs_names = ACTIONS, ACTION_BRANCH, OBS_NAMES
        self.action_space = spaces.Discrete(len(ACTIONS))
        self.reset(seed=0)
        self.observation_space = spaces.Box(-5.0, 5.0, shape=(len(self._obs()),), dtype=np.float32)

    # ---------------------------------------------------------------- episode
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        cfg = self.base_cfg
        if self.scenario_mix:
            cfg = replace(cfg, **SCENARIOS[self.scenario_mix[int(self.np_random.integers(len(self.scenario_mix)))]])
        cfg, key = self._field_cfg(cfg)
        if self.randomize:
            cfg = self._random_conditions(cfg)
        self.cfg = c = cfg
        if key != self._geom_key:
            self._setup_geometry()
            self._geom_key = key
        self._derive()
        self._make_truth()
        # measurement noise is keyed by what is read (see MeasureMixin._rng): this mission's seed, and the salt a
        # fork() uses to draw fresh readings for everything still to come
        self.noise_seed, self.noise_salt = int(self.np_random.integers(2 ** 62)), 0
        self.plant_decisions = 0
        self.prec_n, self.wsum_n = np.zeros(self.N), np.zeros(self.N)
        self.prec_h, self.wsum_h = np.zeros(self.N), np.zeros(self.N)
        self.scan_count = np.zeros(self.N, dtype=int)           # successful shots (plant detected)
        self.shots = np.zeros(self.N, dtype=int)                # all shots, incl. those where YOLO missed
        self.trust, self.surprise = 1.0, 1.0
        self.nug = np.ones(self.N)
        self.anomalies = self.failed_shots = self.nudges_total = 0
        self.mu_n = np.full(self.N, c.prior_ndvi)
        self.mu_h = np.full(self.N, c.prior_height_cm)
        self.sd_n = np.full(self.N, self.sd0)
        self.sd_h = np.full(self.N, math.sqrt(c.map_height_spatial_sd_cm ** 2 + c.map_height_plant_sd_cm ** 2))
        thr = c.ndvi_stress_threshold
        self.pstress = _phi((thr - self.mu_n) / self.sd_n)
        self.blk_mu_n = self.mu_n.reshape(self.NB, self.K).sum(1)
        self.blk_mu_h = self.mu_h.reshape(self.NB, self.K).sum(1)
        self.blk_sd_n = self.sd_n.reshape(self.NB, self.K).sum(1)
        self.blk_count = np.zeros(self.NB, dtype=int)                # plants measured per block
        self.z_mu_sum = np.bincount(self.zone_of, weights=self.mu_n, minlength=self.Z + 1)
        self.z_called = np.bincount(self.zone_of, weights=(self.mu_n < thr).astype(float), minlength=self.Z + 1)
        called = self.mu_n < thr
        self.tp = int((called & self.stressed).sum())
        self.fp = int((called & ~self.stressed & self.vplant).sum())
        vb = self.vblock
        self.blk_score = self._block_scores(np.arange(self.NB))        # per-block accuracy (running sum below)
        self.blk_score_sum = float(self.blk_score[vb].sum())
        self.row, self.pos = 0, int(self.p0[0])
        self.phase, self.block_mode = "block", "quick"       # decision kind now; how the current block is measured
        self.plan, self.plan_k, self.plan_block = [], 0, 0  # the current block's planned plants
        self.block_decisions = self.second_shots = 0
        self.shift_sum, self.shift_n = 0.0, 0                # how far from its stop the robot shot from
        self.batt_wh = self.capacity_wh
        self.sets_left = c.spare_pack_sets
        self.time_s = self.energy_wh = 0.0
        self.swaps = self.forced_returns = self.zaps = 0
        self.n_scanned = self.steps = self.illegal = 0
        self.quick_plants = self.careful_plants = 0                       # plants measured, by mode
        # the robot's own running averages of 0.1-m moves and shots per plant, by mode (used to plan energy)
        self.avg_moves = {"careful": c.careful_prior_nudges, "quick": c.quick_prior_nudges}
        self.avg_shots = {"careful": c.careful_prior_shots, "quick": c.quick_prior_shots}
        # ... and, over ALL plants, what it took to get the first NDVI reading (re-shots when out of the NIR view)
        self.avg_to_read = {"moves": c.quick_prior_nudges, "shots": c.quick_prior_shots}
        self.x = self.x0 = 0.0
        self.best_qd, self.best_qn = np.zeros(self.N), np.zeros(self.N)   # sharpest shot of each plant so far
        self.blk_maxz = np.zeros(self.NB)                                 # most surprising reading per block (sigmas)
        self.zap_timer = 0.0                            # field-work seconds since the last zapper stop
        self.work_s = 0.0
        self.returned = self.stranded = self.complete = self.ended = False
        s, e = self._start_trip()
        self._spend(s, e)
        self._arrive()                                  # at the first plant of the first block
        self.unsampled_proj = self._projected_unsampled()   # blocks projected to stay unsampled
        self.acc = self._accuracy(self.unsampled_proj)      # ACCURACY of the map right now
        return self._obs(), {}

    # ---------------------------------------------------------------- step = one decision
    def step(self, action):
        """One decision. At a block (phase 'block'): how carefully and how many plants. At a plant (phase 'plant'):
        where to shoot from, or DONE. The robot then does everything up to the next decision by itself: shooting,
        moving on to the next planned plant, and after the last one driving on to the next block."""
        c = self.cfg
        acc_before, t0, swaps0 = self.acc, self.time_s, self.swaps
        info = {}
        m = self.action_masks()
        a = int(action)
        if not m[a]:                  # never happens with a masked policy; counted, then made legal
            self.illegal += 1
            a = self._legal_fallback(a, m)
        if self.phase == "block":
            self.block_mode, k = decode(a)
            self.plan, self.plan_k, self.plan_block = self._slots(k), 0, self.blk
            self.block_decisions += 1
            self.phase = "plant"                         # the robot already stands at the block's first plant
        else:
            self._plant_step(a)
        if self.positioner is not None:                  # positioning by the positioner, up to the next block
            self._position_until_block()
        if self.swaps > swaps0:
            info["event"] = "swap"
        self.steps += 1
        terminated = self.ended
        truncated = (not terminated) and self.steps >= c.max_steps
        end = terminated or truncated
        # REWARD = change of the map's ACCURACY - the time it cost (HOUR_VALUE_POINTS per hour), so over a
        # mission the rewards add up to exactly the SCORE it is judged on (minus a constant per field).
        self.unsampled_proj = float(self.blocks_unsampled()) if end else self._projected_unsampled()
        self.acc = self._accuracy(self.unsampled_proj)
        reward = (self.acc - acc_before) - c.hour_value_points * (self.time_s - t0) / 3600.0
        if end:
            self.complete = self.returned and not self.stranded and self.blocks_unsampled() == 0
            if self.stranded:
                reward -= c.stranded_penalty_points
            info["summary"] = self.summary()
        return self._obs(), float(reward), terminated, truncated, info

    def _plant_step(self, a):
        """One decision at a plant: DONE, or SHOOT_x (move there and shoot; a quick plant is left once it has its
        reading, a careful one after two)."""
        if a == DONE:
            self._leave_plant()
        else:
            self._plant_shot(a)
            if not self.ended and self.plant_done():
                self._leave_plant()

    def _position_until_block(self):
        """Let the positioner make every plant decision until the next block decision (or the end)."""
        while not self.ended and self.phase == "plant":
            m = self.action_masks()
            a = int(self.positioner(self))
            if not m[a]:
                self.illegal += 1
                a = self._legal_fallback(a, m)
            self._plant_step(a)
            self.plant_decisions += 1

    def fork(self, salt=None):
        """A copy of the mission right now, to try a choice without touching this one. What never changes during a
        mission is shared, not copied (fast). salt: fresh random readings from here on, the same for every fork with
        that salt (so forks trying different choices see the same noise: common random numbers)."""
        memo = {id(v): v for v in (getattr(self, n, None) for n in self._FIXED) if v is not None}
        e = copy.deepcopy(self, memo)
        if salt is not None:
            e.noise_salt = salt
        return e

    def _leave_plant(self):
        """Done with this plant: on to the next planned plant of the block, or on to the next block."""
        self._end_plant()
        self.plan_k += 1
        if self.ended:
            return
        if self.plan_k < len(self.plan):
            self._drive_to(self.plan[self.plan_k])
            return
        # PROTOCOL: a sample every 10 ft. Only if YOLO misses are enabled can the planned plants all fail - then the
        # robot keeps trying the next plants of the block, carefully.
        if self.blk_count[self.plan_block] == 0 and self.pos % self.K < self.K - 1:
            self.block_mode = "careful"
            self.plan.append(self.pos + 1)
            self._drive_to(self.pos + 1)
            return
        self._next_block()
        if not self.ended:
            self.phase = "block"

    def _legal_fallback(self, a, m):
        """The nearest legal action to an illegal one (only reached by an unmasked policy)."""
        if self.phase == "block":
            if is_block_action(a):
                mode, k = decode(a)
                legal = [kk for kk in range(k, 0, -1) if m[encode(mode, kk)]]
                if legal:
                    return encode(mode, legal[0])
            return A["QUICK_1"]
        return SHOOT_HERE if m[SHOOT_HERE] else int(np.flatnonzero(m)[0])

    # ---------------------------------------------------------------- reporting
    def summary(self):
        c, Z, vp, vb = self.cfg, self.Z, self.vplant, self.vblock
        thr = c.ndvi_stress_threshold
        measured = self.scan_count > 0
        called = (self.mu_n < thr) & vp
        st = self.stressed
        tp, fp = int((called & st).sum()), int((called & ~st).sum())
        est = self.z_mu_sum[:Z] / self.zone_size[:Z]
        ztrue = self.z_flag_true[:Z]
        pflag = self.z_called[:Z] / self.zone_size[:Z] >= c.part_stress_frac
        ptp, pfp = int((pflag & ztrue).sum()), int((pflag & ~ztrue).sum())
        sampled = (self.blk_count > 0) & vb
        n_meas = int(measured.sum())
        unsampled = self.blocks_unsampled()
        accuracy = self._accuracy(unsampled)
        parts = {f"acc_{k}": 100.0 * v for k, (_, v) in self._accuracy_terms(unsampled).items()}
        hours = self.time_s / 3600.0
        return dict(
            score=0.0 if self.stranded else accuracy - c.hour_value_points * hours, accuracy=accuracy, **parts,
            field=c.field, acres=c.field_acres, blocks=self.NBv, blocks_sampled=float(sampled.sum() / self.NBv),
            plants_measured=n_meas,
            plants_per_block=float(self.blk_count[sampled].mean()) if sampled.any() else 0.0,
            plants_quick=self.quick_plants, plants_careful=self.careful_plants,
            shots_per_plant=float(self.shots[measured].sum() / max(n_meas, 1)),
            nudges_per_plant=float(self.nudges_total / max(n_meas, 1)),
            second_shot_share=float(self.second_shots / max(n_meas, 1)),
            mean_shot_shift_cm=float(100 * self.shift_sum / max(self.shift_n, 1)),
            decisions=self.steps, block_decisions=self.block_decisions, plant_decisions=self.plant_decisions,
            yolo_misses=int(self.failed_shots),
            block_rmse_ndvi=float(np.sqrt(np.mean((self.blk_mu_n / self.K - self.blk_true_n)[vb] ** 2))),
            block_rmse_height_cm=float(np.sqrt(np.mean((self.blk_mu_h / self.K - self.blk_true_h)[vb] ** 2))),
            stressed_total=int(st.sum()), recall=tp / max(int(st.sum()), 1),
            f1=_f1(tp, fp, int(st.sum()) - tp), false_alarms=fp,
            ndvi_rmse=float(np.sqrt(np.mean((self.mu_n - self.ndvi)[vp] ** 2))),
            part_rmse=float(np.sqrt(np.mean((est - self.z_true_mean[:Z]) ** 2))),
            part_f1=_f1(ptp, pfp, int(ztrue.sum()) - ptp),
            parts_needing_attention=int(ztrue.sum()),
            swaps=self.swaps, zaps=self.zaps, hours=hours, energy_wh=self.energy_wh,
            complete=bool(self.complete), stranded=bool(self.stranded), trust=self.trust,
        )

    def block_maps(self):
        """Two text maps, one line per pass (pass 0 = next to base), one character per 10-ft block:
        plants measured in the block, and whether the block's stress was found."""
        thr = self.cfg.ndvi_stress_threshold
        called = ((self.mu_n < thr) & self.vplant).reshape(self.NB, self.K).any(1)
        truth = self.stressed.reshape(self.NB, self.K).any(1)
        counts, status = [], []
        for r in range(self.R):
            bs = range(r * self.B, (r + 1) * self.B)
            counts.append(f"pass {r:2d} |" + "".join((str(self.blk_count[b]) if self.blk_count[b] else "-")
                                                     if self.vblock[b] else " " for b in bs) + "|")
            status.append(f"pass {r:2d} |" + "".join(" " if not self.vblock[b] else
                                                     ("#" if called[b] else "!") if truth[b] else ("?" if called[b] else ".")
                                                     for b in bs) + "|")
        return "\n".join(counts), "\n".join(status)
