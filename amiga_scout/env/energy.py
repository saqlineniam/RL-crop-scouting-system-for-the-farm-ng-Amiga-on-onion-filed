"""Time and energy: power draw, the bug zapper, battery trips and swaps, the protocol energy plan and the
masks that keep the robot from ever stranding or skipping a block."""
from __future__ import annotations

import math

import numpy as np

from ..actions import A, ACTIONS, DONE, FIRST_SHOOT, SHOOT_HERE, encode, shoot_offset


class EnergyMixin:
    """Time, energy, batteries and the action masks (used by AmigaMissionEnv)."""

    def _derive(self):
        """Values computed from the settings (recomputed when conditions change between episodes)."""
        c = self.cfg
        self.capacity_wh = c.pack_voltage_v * c.pack_capacity_ah * c.packs_onboard * c.usable_capacity_frac
        self.reserve_wh = c.return_trigger_pct / 100.0 * self.capacity_wh
        self.zap_every_s = c.zap_every_min * 60.0
        self.zap_s = c.zap_duration_min * 60.0 if c.zapper_enabled else 0.0
        self.zap_share = self.zap_s / self.zap_every_s if self.zap_s > 0 else 0.0   # 15 min per 30 min = 0.5
        self.t_nudge = c.nudge_step_m / c.nudge_speed_mps + 0.5
        self.t_stop = c.hop_overhead_s + c.stop_settle_s              # slow down, stop, let the camera settle
        # speeds: stop-and-go inside a bed while scouting, and faster when driving without stopping (headland,
        # trips home); never above the platform's top speed
        self.v_row = min(c.row_speed_mps, c.max_speed_mps)
        self.v_travel = min(c.travel_speed_mps, c.max_speed_mps)
        self.t_drive_block = self.block_m / self.v_row                # driving through one 10-ft block
        # turn from the end of pass r into pass r+1 (on a real field the passes end at different places, so
        # the robot also drives along the headland by the difference); suffix sums = all turns still ahead
        turns = [self._est(c.row_turn_s + (self.dy + 2 * c.headland_m + self._turn_shift(r)) / self.v_travel, 0.0)
                 for r in range(self.R - 1)]
        tw = np.array([t for t, _ in turns] + [0.0]); te = np.array([e for _, e in turns] + [0.0])
        self.turn_rest_s, self.turn_rest_wh = np.cumsum(tw[::-1])[::-1], np.cumsum(te[::-1])[::-1]

    def _est(self, drive_s, stand_s):
        """(wall s, Wh) of field work: driving + standing, scan light on, plus the bug zapper's share."""
        c = self.cfg
        work = drive_s + stand_s
        wh = (drive_s * (c.hotel_power_w + c.drive_power_w + c.light_power_w)
              + stand_s * (c.hotel_power_w + c.light_power_w)
              + work * self.zap_share * (c.hotel_power_w + c.zapper_power_w)) / 3600.0
        return work * (1.0 + self.zap_share), wh

    def _drive_power(self):
        """W while driving: motors + computer/cameras + the scan light (on except during bug-zapper stops)."""
        c = self.cfg
        return c.drive_power_w + c.hotel_power_w + c.light_power_w

    def _zap_due(self, secs):
        """Would doing `secs` of field work trigger a bug-zapper stop?"""
        return self.zap_s > 0 and self.zap_timer + secs >= self.zap_every_s

    def _zap_cost(self):
        """(seconds, Wh) of a bug-zapper stop: the scan light is off (the zapper has its own light)."""
        c = self.cfg
        return self.zap_s, self.zap_s * (c.hotel_power_w + c.zapper_power_w) / 3600.0

    def _spend(self, secs, wh):
        self.time_s += secs
        self.batt_wh -= wh
        self.energy_wh += wh

    def _work(self, secs, driving):
        """Field work (scan light on). BUG ZAPPER: after every zap_every_min of field work the platform
        stops where it is and zaps for zap_duration_min (time + hotel/zapper power, no measuring)."""
        c = self.cfg
        power = c.hotel_power_w + c.light_power_w + (c.drive_power_w if driving else 0.0)
        self._spend(secs, secs * power / 3600.0)
        self.zap_timer += secs
        self.work_s += secs                             # all field work so far (for smooth time values)
        if self.zap_s > 0 and self.zap_timer >= self.zap_every_s:
            self._spend(*self._zap_cost())
            self.zaps += 1
            self.zap_timer = 0.0
        if self.batt_wh < -1e-6:
            self.stranded = self.ended = True

    def _battery_trip(self):
        """The packs reached the return level: drive straight on to the end of the pass, follow the headland
        home, swap both packs, come back the same way and resume at this plant (the lab procedure). Returns
        False when no charged pair is left - then the robot stays home and the rest of the route is not sampled
        (the battery budget in action_masks keeps this from happening)."""
        c = self.cfg
        s, e = self._trip(self.row, self.pos)
        self._spend(s, e)
        self.forced_returns += 1
        if self.batt_wh < -1e-6:
            self.stranded = self.ended = True
            return False
        if self.sets_left <= 0:
            self.returned = self.ended = True
            return False
        self.sets_left -= 1
        self.swaps += 1
        self.batt_wh = self.capacity_wh
        self.time_s += c.swap_time_s
        self._spend(s, e)
        self._arrive(resume=True)
        return True

    def energy_available(self):
        """Wh the robot can still use: this charge above the return level plus the spare pairs'."""
        return (self.batt_wh - self.reserve_wh) + self.sets_left * (self.capacity_wh - self.reserve_wh)

    def block_cost(self, mode, k):
        """(wall s incl. the zapper's share, Wh) of a block with k plants measured this way, driving on
        included. Every plant is costed with the robot's own running averages of moves and shots for this mode, so
        re-shots (a plant outside the NIR view) and its positioning are planned for - also for the protocol plant."""
        c = self.cfg
        shot = c.shot_s + (c.careful_average_s if mode == "careful" else 0.0)
        moves, shots = self.avg_moves[mode], self.avg_shots[mode]
        if mode == "quick":                              # at least what the first reading takes (any mode's plants)
            moves, shots = max(moves, self.avg_to_read["moves"]), max(shots, self.avg_to_read["shots"])
        drive = self.t_drive_block + k * (self.t_stop + moves * self.t_nudge)
        return self._est(drive, k * shots * shot)

    def protocol_plan(self, from_start=False):
        """(wall s, Wh) to finish the route at the protocol minimum (one quick plant per block) from the block
        the robot is at: the blocks, the pass turns, the trip back after every battery swap this needs (the
        trip out uses the reserve), and the trip home."""
        c = self.cfg
        n = self.NBv if from_start else self.blocks_left() + 1
        r = 0 if from_start else self.row
        t1, e1 = self.block_cost("quick", 1)
        home_s = self._worst_home_s(self.R - 1)
        home_wh = home_s * self._drive_power() / 3600.0
        wall = n * t1 + self.turn_rest_s[r] + home_s
        wh = n * e1 + self.turn_rest_wh[r] + home_wh
        usable = self.capacity_wh - self.reserve_wh
        need = wh - (usable if from_start else self.batt_wh - self.reserve_wh)
        if need > 0:
            swaps = math.ceil(need / usable)
            wall += swaps * (c.swap_time_s + 2 * home_s)
            wh += swaps * home_wh
        return wall, wh

    def spare_wh(self):
        """Energy left beyond what the protocol for the rest of the route needs (after the safety margin)."""
        m = 1.0 + self.cfg.budget_margin
        return (self.energy_available() - m * self.protocol_plan()[1]) / m

    def _projected_unsampled(self):
        """Blocks ahead that the energy left cannot reach even at the protocol minimum - charged in the reward
        as soon as that becomes likely, not only at the end (with the budget masks this stays 0)."""
        if self.ended:
            return 0.0
        need = self.protocol_plan()[1]
        avail = max(self.energy_available(), 0.0)
        return 0.0 if avail >= need else (self.blocks_left() + 1) * (1.0 - avail / need)

    def spare_plants_per_block(self):
        """Spare energy, in careful extra plants per block still to do (incl. this one)."""
        extra = self.block_cost("careful", 2)[1] - self.block_cost("careful", 1)[1]
        return self.spare_wh() / extra / max(self.blocks_left() + 1, 1)

    # ---------------------------------------------------------------- masks (hard rules live here)
    def action_masks(self):
        """At a block: QUICK_1 (the protocol minimum) is always allowed; more plants or more care only while the
        energy left, spare packs included, still covers one plant in every block ahead with a margin. At a plant:
        shooting from where the robot stands is always allowed for the first shot; moving, and a second shot, only
        within the same energy budget; DONE only once the plant has been measured. So neither extra care nor the
        robot's positioning can ever cost a 10-ft sample, and the robot is never stranded."""
        m = np.zeros(len(ACTIONS), dtype=bool)
        if self.ended:
            m[A["QUICK_1"]] = True
            return m
        spare = self.spare_wh()
        if self.phase == "block":
            m[A["QUICK_1"]] = True
            base = self.block_cost("quick", 1)[1]
            for mode in ("quick", "careful"):
                for k in range(1, self.K + 1):
                    if self.block_cost(mode, k)[1] - base <= spare:
                        m[encode(mode, k)] = True
            return m
        c = self.cfg
        shot_wh = self._est(0.0, c.shot_s + (c.careful_average_s if self.block_mode == "careful" else 0.0))[1]
        for a in range(FIRST_SHOOT, DONE):
            extra = self.move_cost_wh(shoot_offset(a)) + (shot_wh if self.plant_read else 0.0)
            m[a] = extra <= max(spare, 0.0)
        if not self.plant_read:
            m[SHOOT_HERE] = True                         # the protocol plant can always be shot where it stopped
        m[DONE] = self.plant_read
        return m
