"""The field and the fixed route: which beds the passes take, where each pass starts and ends on a real
outline, the parts, travel along the route and the trips home."""
from __future__ import annotations

from dataclasses import replace
import math

import numpy as np

from ..config import BARE_COVER_FRAC, GAP_COVER_FRAC, LEAF_WIDTH_MM, STAGE_HEIGHT_CM
from ..fields import draw_piece, field_map, parse_name, region_bounds, square_field, window_acres


class RouteMixin:
    """Geometry of the field and the route (used by AmigaMissionEnv)."""

    def _field_cfg(self, cfg):
        """Draw this mission's field from the pool (and, on a real field, its route); returns the cfg for it."""
        name = self.pool[int(self.np_random.integers(len(self.pool)))] if len(self.pool) > 1 else self.pool[0]
        if name.startswith("sq"):
            return replace(cfg, **square_field(float(name[2:]))), (name,)
        f = field_map(name)
        start = cfg.route_start_bed if cfg.route_start_bed >= 0 else int(self.np_random.integers(cfg.pass_every_n_beds))
        corner = cfg.route_corner if cfg.route_corner >= 0 else int(self.np_random.integers(4))
        h = STAGE_HEIGHT_CM.get(f["date"], cfg.height_mean_cm)
        leaf = LEAF_WIDTH_MM.get(f["date"], cfg.leaf_width_mm)       # onion leaves widen as they grow
        acres, self.piece = f["acres"], None
        _, region, size = parse_name(name)
        bounds = region_bounds(name)                 # the training or test region of the field (80/20 split)
        if size:                                     # a random piece (of the region): a new one every mission
            *self.piece, acres = draw_piece(f, size, self.np_random, bounds)
            self.piece = tuple(self.piece)
        elif bounds:
            self.piece, acres = tuple(bounds), window_acres(f, *bounds)
        return (replace(cfg, field=name, field_acres=acres, height_mean_cm=h, prior_height_cm=h, leaf_width_mm=leaf,
                        route_start_bed=start % cfg.pass_every_n_beds, route_corner=corner % 4),
                (name, start % cfg.pass_every_n_beds, corner % 4, cfg.pass_every_n_beds, self.piece))

    def _setup_geometry(self):
        """Route, blocks, parts and map windows for the current field (real units).
        "rows" below are the robot's PASSES; "cols" are PLANTS along a pass, grouped in 10-ft blocks. A real field
        is not a rectangle: it is laid on a rectangular grid of passes x plants and every pass has its own start
        and end (vplant / vblock = inside the crop; p0 / p1 = first / last travel position of each pass)."""
        c = self.cfg
        self.K = c.plants_per_block
        self.block_m = c.block_ft * 0.3048
        self.dxp = self.block_m / self.K                           # plant spacing along a pass (0.508 m)
        self.real = not c.field.startswith("sq")
        if self.real:
            self._real_layout(field_map(c.field))
        else:
            self.dy = c.pass_every_n_beds * c.bed_spacing_m
            self.width_m = c.field_acres * 4046.86 / c.row_length_m
            self.R = max(1, int(self.width_m // self.dy))          # passes
            self.B = max(1, int(c.row_length_m // self.block_m))   # 10-ft blocks per pass
            self.C = self.B * self.K                               # plants per pass
            self.c0, self.c1 = np.zeros(self.R, int), np.full(self.R, self.C - 1)
            self.vplant = np.ones(self.R * self.C, bool)
        self.N, self.NB = self.R * self.C, self.R * self.B
        rr, cc = np.divmod(np.arange(self.N), self.C)
        self.block_of = rr * self.B + cc // self.K                 # block b holds plants b*K .. b*K+K-1
        self.vblock = self.vplant.reshape(self.NB, self.K).any(1)
        self.Nv, self.NBv = int(self.vplant.sum()), int(self.vblock.sum())
        # travel positions of each pass (the route snakes: even passes run with the grid, odd ones against it)
        odd = np.arange(self.R) % 2 == 1
        self.p0 = np.where(odd, self.C - 1 - self.c1, self.c0)
        self.p1 = np.where(odd, self.C - 1 - self.c0, self.c1)
        self.route_rank = np.full(self.NB, -1)                     # order of every block along the route
        k = 0
        for r in range(self.R):
            for pos in range(int(self.p0[r]), int(self.p1[r]) + 1, self.K):
                self.route_rank[r * self.B + self._col(r, pos) // self.K] = k
                k += 1
        # parts: POINTS_PER_30_ACRES per 30 acres, near-square pieces of the field (only pieces with crop count)
        n_parts = max(1, round(c.points_per_30_acres * c.field_acres / 30.0))
        x, y = (cc + 0.5) * self.dxp, (rr + 0.5) * self.dy
        side = math.sqrt(c.field_acres * 4046.86 / n_parts)
        for _ in range(30):                                        # grow/shrink the pieces until the count is right
            ix, iy = (x // side).astype(int), (y // side).astype(int)
            key = iy * (int(x.max() // side) + 1) + ix
            used = np.unique(key[self.vplant])
            if abs(len(used) - n_parts) <= max(1, n_parts // 20):
                break
            side *= math.sqrt(len(used) / n_parts)
        self.Z = len(used)
        self.zone_of = np.full(self.N, self.Z)                     # zone Z = outside the crop (not scored)
        self.zone_of[self.vplant] = np.searchsorted(used, key[self.vplant])
        self.zone_size = np.bincount(self.zone_of, minlength=self.Z + 1).astype(float)
        self.zone_size[self.Z] = max(self.zone_size[self.Z], 1.0)
        self.zx, self.zy = int(x.max() // side) + 1, int(y.max() // side) + 1
        # map model windows, in 10-ft blocks: J = blocks refreshed after a shot, S = measured blocks used for it
        L = c.map_length_m
        self.sd0 = math.sqrt(c.map_ndvi_spatial_sd ** 2 + c.map_ndvi_plant_sd ** 2)
        self.wbJ, self.wrJ = math.ceil(2.0 * L / self.block_m), math.ceil(2.0 * L / self.dy)
        self.wbS, self.wrS = math.ceil(3.0 * L / self.block_m), math.ceil(3.0 * L / self.dy)
        dmax = 2 * max(self.wbS, self.wrS) + 2
        self.kblk = np.exp(-0.5 * (np.arange(dmax) * self.block_m / L) ** 2)
        self.krow = np.exp(-0.5 * (np.arange(dmax) * self.dy / L) ** 2)

    def _real_layout(self, f):
        """Lay the robot's fixed route on a real field: one pass every pass_every_n_beds beds, the first on bed
        route_start_bed, starting from corner route_corner; each pass runs from where the crop starts to where it
        ends on that bed (field margins and bare ends are not crop). Fills bg_leaf / bg_cover (the drone's
        plant-pixel NDVI and canopy cover at every plant spot of the route) and the gap flags (spots with almost no
        canopy: the robot finds only a small, weak plant there)."""
        c, K = self.cfg, self.K
        leaf, cover, good = f["leaf"], f["cover"], f["good"]
        if getattr(self, "piece", None):           # a piece of the field: only its beds and its stretch of them
            k0, k1, j0, j1 = self.piece
            leaf, cover, good = leaf[k0:k1, j0:j1], cover[k0:k1, j0:j1], good[k0:k1, j0:j1]
        if c.route_corner & 1:                     # start from the other side of the field (last bed first)
            leaf, cover, good = leaf[::-1], cover[::-1], good[::-1]
        if c.route_corner & 2:                     # start from the other end of the beds
            leaf, cover, good = leaf[:, ::-1], cover[:, ::-1], good[:, ::-1]
        nb = leaf.shape[1] // K
        passes = []
        for k in range(c.route_start_bed, leaf.shape[0], c.pass_every_n_beds):
            g = good[k, :nb * K].reshape(nb, K)
            cv = np.where(g, cover[k, :nb * K].reshape(nb, K), 0.0).sum(1) / np.maximum(g.sum(1), 1)
            crop = np.flatnonzero((g.sum(1) >= K - 1) & (cv >= BARE_COVER_FRAC * f["cover_median"]))
            # the pass runs from the first to the last crop block of the bed: the robot drives the whole bed, so a
            # bare stretch inside it (a stand gap, 3-9 m on these fields) is sampled too and reads as gap plants
            if len(crop):                                  # even a 1-block sliver at a corner is sampled (protocol)
                passes.append((k, int(crop[0]), int(crop[-1])))
        if not passes:
            raise SystemExit(f"{c.field}: no crop found along the beds")
        bmin, bmax = min(p[1] for p in passes), max(p[2] for p in passes)
        self.R, self.B = len(passes), bmax - bmin + 1
        self.C = self.B * K
        self.dy = c.pass_every_n_beds * f["bed_spacing_m"]
        self.width_m = self.R * self.dy
        self.c0 = np.array([(p[1] - bmin) * K for p in passes])
        self.c1 = np.array([(p[2] - bmin + 1) * K - 1 for p in passes])
        self.vplant = np.zeros(self.R * self.C, bool)
        self.bg_leaf = np.full(self.R * self.C, np.nan)
        self.bg_cover = np.zeros(self.R * self.C)
        for r, (k, b0, b1) in enumerate(passes):
            js = slice(b0 * K, (b1 + 1) * K)
            lf, cv, g = leaf[k, js].reshape(-1, K), cover[k, js].reshape(-1, K), good[k, js].reshape(-1, K)
            ok = g & np.isfinite(lf)
            n_ok = ok.sum(1, keepdims=True)
            fill_l = np.where(n_ok > 0, np.where(ok, lf, 0.0).sum(1, keepdims=True) / np.maximum(n_ok, 1), np.nan)
            fill_c = np.where(g, cv, 0.0).sum(1, keepdims=True) / np.maximum(g.sum(1, keepdims=True), 1)
            # a spot outside the map (edge pixel, no data) takes its block's mean; a mapped spot without leaves is a gap
            lf = np.where(ok, lf, np.where(g, np.nan, fill_l))
            cv = np.where(g, cv, fill_c)
            i0 = r * self.C + self.c0[r]
            self.bg_leaf[i0:i0 + lf.size] = lf.ravel()
            self.bg_cover[i0:i0 + cv.size] = cv.ravel()
            self.vplant[i0:i0 + lf.size] = True
        self.gap = self.vplant & ((self.bg_cover < GAP_COVER_FRAC * f["cover_median"]) | ~np.isfinite(self.bg_leaf))
        self.map_info = dict(beds=leaf.shape[0], passes=[p[0] for p in passes], spans=passes)

    def _turn_shift(self, r):
        """Metres along the headland between the end of pass r and the start of pass r+1."""
        if r % 2 == 0:                                   # even passes end at the high end, odd ones start there
            return abs(int(self.c1[r]) - int(self.c1[r + 1])) * self.dxp
        return abs(int(self.c0[r]) - int(self.c0[r + 1])) * self.dxp

    # ---------------------------------------------------------------- geometry
    def _col(self, row, pos):
        return pos if row % 2 == 0 else self.C - 1 - pos

    @property
    def here(self):
        return self.row * self.C + self._col(self.row, self.pos)

    @property
    def blk(self):
        return int(self.block_of[self.here])

    def _home_dist(self, row, pos):
        """Metres from this plant to base, the way the Amiga really leaves a pass: it cannot turn in the
        crop, so it drives STRAIGHT ON to the end of the pass it is in, leaves the field onto the
        headland, and follows the headland to base (at the pass-0 / west corner). Coming back after a
        swap it drives the same way in reverse and resumes at this plant. Even passes run west->east."""
        c = self.cfg
        in_row = (int(self.p1[row]) - pos) * self.dxp + c.headland_m
        y = (row + 0.5) * self.dy
        x_exit = (int(self.c1[row]) + 1) * self.dxp if row % 2 == 0 else int(self.c0[row]) * self.dxp
        around = abs(x_exit - int(self.c0[0]) * self.dxp) + (2 * c.headland_m if row % 2 == 0 else 0.0)
        return in_row + y + around + c.base_to_field_m   # base: next to where pass 0 starts

    def _trip(self, row, pos):
        """(seconds, Wh) to drive between base and this plant, at travel speed (scan light on - it always is)."""
        s = self._home_dist(row, pos) / self.v_travel
        return s, s * self._drive_power() / 3600.0

    def _start_trip(self):
        c = self.cfg
        s = (c.base_to_field_m + c.headland_m + 0.5 * self.dy) / self.v_travel
        return s, s * self._drive_power() / 3600.0

    def _worst_home_s(self, row):
        c = self.cfg
        return ((row + 0.5) * self.dy + 2 * self.C * self.dxp + 3 * c.headland_m + c.base_to_field_m) / self.v_travel

    def _drive_to(self, pos):
        """Drive along this pass to travel position `pos` (stop-and-go, at in-bed speed) and stop there."""
        self._work((pos - self.pos) * self.dxp / self.v_row + self.t_stop, True)
        self.pos = pos
        self._arrive()

    def _next_block(self):
        """Drive on to the first plant of the next 10-ft block (turning into the next pass at the end of a
        pass); after the last block of the route, drive home."""
        c = self.cfg
        start = self.pos - self.pos % self.K
        if start + self.K <= int(self.p1[self.row]):
            self._drive_to(start + self.K)
        elif self.row < self.R - 1:
            rest = (int(self.p1[self.row]) - self.pos) * self.dxp
            self._work(c.row_turn_s + (rest + 2 * c.headland_m + self.dy + self._turn_shift(self.row))
                       / self.v_travel + c.stop_settle_s, True)
            self.row, self.pos = self.row + 1, int(self.p0[self.row + 1])
            self._arrive()
        else:
            self._go_home()

    def _go_home(self):
        s, e = self._trip(self.row, self.pos)
        self._spend(s, e)
        if self.batt_wh < -1e-6:
            self.stranded = True
        self.returned = self.ended = True

    def _slots(self, k):
        """Travel positions of the k plants to measure in this block: its first plant (where the robot is) and
        the others spread evenly over the rest of the block."""
        start, n = self.pos, self.K - 1
        return [start] + [start + 1 + int((j + 0.5) * n / (k - 1)) for j in range(k - 1)]

    # ---------------------------------------------------------------- battery budget (planning)
    def blocks_left(self):
        """10-ft blocks after the current one (crop blocks along the route)."""
        return self.NBv - 1 - int(self.route_rank[self.blk])

    def blocks_unsampled(self):
        return int(((self.blk_count == 0) & self.vblock).sum())

    # ---------------------------------------------------------------- observation
    def _block_at(self, pos):
        """Block id at a travel position on the current pass (None if off the pass)."""
        if pos < int(self.p0[self.row]) or pos > int(self.p1[self.row]):
            return None
        return int(self.block_of[self.row * self.C + self._col(self.row, pos)])
