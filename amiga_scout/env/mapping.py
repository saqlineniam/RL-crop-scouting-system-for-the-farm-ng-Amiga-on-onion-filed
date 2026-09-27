"""The robot's own crop map (block-level Gaussian process with TRUST and ANOMALY) and the ACCURACY it is
scored on (block values, stressed plants found, parts, protocol)."""
from __future__ import annotations

import numpy as np

from ..config import (BLOCK_TOL_CM, BLOCK_TOL_NDVI, PART_TOL_NDVI, SCORE_W_BLOCKS, SCORE_W_DETECTION,
                      SCORE_W_PARTS, SCORE_W_PROTOCOL)
from ..mathutil import _phi


class MapMixin:
    """The robot's map and its accuracy (used by AmigaMissionEnv)."""

    # ---------------------------------------------------------------- map (belief) and its accuracy
    @staticmethod
    def _flag_score(tp, fp, n_true, floor):
        """1 = every stressed plant (or part) found and no false alarms. Each miss costs 1/max(n_true, floor),
        each false alarm half of that - so on a field with almost no stress, one false alarm costs a little,
        not everything (an F1 score would jump from 1 to 0). Straight in the counts, not cut off at 0: every find
        and every false alarm counts the moment it happens (a floor at 0 made early finds worth nothing while
        false alarms outnumbered them)."""
        return 1.0 - ((n_true - tp) + 0.5 * fp) / max(n_true, floor)

    def _block_scores(self, bs):
        """Score 0-1 of each 10-ft block in bs: half its NDVI, half its height; a value off by the tolerance
        (BLOCK_TOL_NDVI, BLOCK_TOL_CM) or more scores 0 (each block on its own, so each counts when it is measured)."""
        K = self.K
        en = np.abs(self.blk_mu_n[bs] / K - self.blk_true_n[bs]) / BLOCK_TOL_NDVI
        eh = np.abs(self.blk_mu_h[bs] / K - self.blk_true_h[bs]) / BLOCK_TOL_CM
        return 0.5 * np.clip(1.0 - en, 0.0, 1.0) + 0.5 * np.clip(1.0 - eh, 0.0, 1.0)

    def _part_terms(self):
        """(mean score 0-1 of the parts' mean NDVI, flag score of the parts that need attention)."""
        c, Z = self.cfg, self.Z
        est = self.z_mu_sum[:Z] / self.zone_size[:Z]
        flag = self.z_called[:Z] / self.zone_size[:Z] >= c.part_stress_frac
        truth = self.z_flag_true[:Z]
        tp = int((flag & truth).sum())
        fp = int((flag & ~truth).sum())
        ndvi = float(np.mean(np.clip(1.0 - np.abs(est - self.z_true_mean[:Z]) / PART_TOL_NDVI, 0.0, 1.0)))
        return ndvi, self._flag_score(tp, fp, int(truth.sum()), max(2.0, 0.1 * Z))

    def _accuracy(self, unsampled):
        """ACCURACY 0-100 of the map right now (uses the hidden truth - scoring only): the lab's 10-ft block
        values, the stressed plants found, the parts, and the protocol (blocks that have / will get a sample).
        The robot is rewarded for every change of this number, minus the time the change cost.
        Only the crop counts (vplant / vblock): a real field's margins are not part of the map."""
        terms = self._accuracy_terms(unsampled)
        return 100.0 * sum(w * v for w, v in terms.values()) / sum(w for w, _ in terms.values())

    def _accuracy_terms(self, unsampled):
        """The four parts of ACCURACY: {name: (weight, value; 1 = perfect)}. Each is an average over blocks, plants
        or parts, so a measurement changes the score as soon as it is made, by what it did to the blocks, plants
        and parts it touched (not through one error of the whole field)."""
        part_ndvi, part_flags = self._part_terms()
        return dict(blocks=(SCORE_W_BLOCKS, self.blk_score_sum / self.NBv),
                    stress=(SCORE_W_DETECTION, self._flag_score(self.tp, self.fp, self.n_stressed, 0.02 * self.Nv)),
                    parts=(SCORE_W_PARTS, 0.5 * part_flags + 0.5 * part_ndvi),
                    protocol=(SCORE_W_PROTOCOL, 1.0 - unsampled / self.NBv))

    def _refresh_map(self, r0, c0):
        """Local Gaussian-process update around pass r0, plant column c0. The slowly varying field is modelled
        per 10-ft block (its plants share it: it changes over ~map_length_m, a block is 3 m) and every plant has
        its own deviation on top, loosened for anomalies. Given the field, a plant's readings only tell about
        that plant - so the update is exact for this model and needs one small solve (blocks, not plants)."""
        cfg, K, B = self.cfg, self.K, self.B
        b0 = c0 // K
        rJ = np.arange(max(0, r0 - self.wrJ), min(self.R, r0 + self.wrJ + 1))
        cJ = np.arange(max(0, b0 - self.wbJ), min(B, b0 + self.wbJ + 1))
        rS = np.arange(max(0, r0 - self.wrS), min(self.R, r0 + self.wrS + 1))
        cS = np.arange(max(0, b0 - self.wbS), min(B, b0 + self.wbS + 1))
        jr, jc = (a.ravel() for a in np.meshgrid(rJ, cJ, indexing="ij"))
        sr, sc = (a.ravel() for a in np.meshgrid(rS, cS, indexing="ij"))
        keep = self.blk_count[sr * B + sc] > 0                       # blocks with at least one measured plant
        sr, sc = sr[keep], sc[keep]
        bs = jr * B + jc                                             # blocks refreshed
        J = (bs[:, None] * K + np.arange(K)).ravel()                 # ... and their plants
        SP = (sr * B + sc)[:, None] * K + np.arange(K)               # plants of the measured blocks
        old_n, old_h, old_sd = self.mu_n[J].copy(), self.mu_h[J].copy(), self.sd_n[J].copy()
        t = self.trust
        chans = (("n", cfg.prior_ndvi, cfg.map_ndvi_spatial_sd ** 2, cfg.map_ndvi_plant_sd ** 2),
                 ("h", cfg.prior_height_cm, cfg.map_height_spatial_sd_cm ** 2, cfg.map_height_plant_sd_cm ** 2))
        if len(sr):
            kss = self.krow[np.abs(sr[:, None] - sr[None, :])] * self.kblk[np.abs(sc[:, None] - sc[None, :])]
            kjs = self.krow[np.abs(jr[:, None] - sr[None, :])] * self.kblk[np.abs(jc[:, None] - sc[None, :])]
        for ch, m0, vs, vi in chans:
            prec_all, wsum_all = getattr(self, f"prec_{ch}"), getattr(self, f"wsum_{ch}")
            # blocks with a reading of THIS channel (a shot can give NDVI but no height - plant closer than the
            # D455's minimum distance - or height but no NDVI - plant outside the NIR camera's view)
            has = (prec_all[SP] > 0).any(1) if len(sr) else np.zeros(0, dtype=bool)
            if has.any():
                # each measured block's field reading: its plants' fused readings, each weighted by 1 / (its own
                # deviation variance + its reading noise)
                SPc = SP[has]
                prec = prec_all[SPc]
                meas = prec > 0
                p1 = np.where(meas, prec, 1.0)
                w = np.where(meas, 1.0 / (t * vi * self.nug[SPc] + 1.0 / p1), 0.0)
                W = w.sum(1)
                Y = (w * wsum_all[SPc] / p1).sum(1) / W
                Kjs = t * vs * kjs[:, has]
                sol = np.linalg.solve(t * vs * kss[np.ix_(has, has)] + np.diag(1.0 / W),
                                      np.column_stack([Y - m0, Kjs.T]))
                fm = m0 + Kjs @ sol[:, 0]
                fv = np.clip(t * vs - np.einsum("ij,ji->i", Kjs, sol[:, 1:]), 1e-10, None)
            else:
                fm, fv = np.full(len(bs), m0), np.full(len(bs), t * vs)
            # each plant = its block's field + its own deviation; a measured plant's readings pull it towards them
            precJ = prec_all[J].reshape(-1, K)
            measJ = precJ > 0
            pJ = np.where(measJ, precJ, 1.0)
            s2 = t * vi * self.nug[J].reshape(-1, K)
            a = np.where(measJ, s2 / (s2 + 1.0 / pJ), 0.0)
            ybar = np.where(measJ, wsum_all[J].reshape(-1, K) / pJ, 0.0)
            getattr(self, f"mu_{ch}")[J] = ((1.0 - a) * fm[:, None] + a * ybar).ravel()
            var = (1.0 - a) ** 2 * fv[:, None] + np.where(measJ, a / pJ, s2)
            getattr(self, f"sd_{ch}")[J] = np.sqrt(var).ravel()
        thr = cfg.ndvi_stress_threshold
        self.pstress[J] = _phi((thr - self.mu_n[J]) / self.sd_n[J])
        bJ, zJ = self.block_of[J], self.zone_of[J]
        np.add.at(self.blk_mu_n, bJ, self.mu_n[J] - old_n)
        np.add.at(self.blk_mu_h, bJ, self.mu_h[J] - old_h)
        np.add.at(self.blk_sd_n, bJ, self.sd_n[J] - old_sd)
        new_called, old_called = self.mu_n[J] < thr, old_n < thr
        np.add.at(self.z_mu_sum, zJ, self.mu_n[J] - old_n)
        np.add.at(self.z_called, zJ, new_called.astype(float) - old_called)
        # running totals behind ACCURACY: stress calls and the 10-ft block errors (crop only)
        s, fa = self.stressed[J], ~self.stressed[J] & self.vplant[J]
        self.tp += int((new_called & s).sum()) - int((old_called & s).sum())
        self.fp += int((new_called & fa).sum()) - int((old_called & fa).sum())
        vb = self.vblock[bs]
        new_score = self._block_scores(bs)
        self.blk_score_sum += float(np.sum((new_score - self.blk_score[bs])[vb]))
        self.blk_score[bs] = new_score

    def block_readings(self, b):
        """Fused NDVI readings of the measured plants of block b."""
        bp = slice(b * self.K, b * self.K + self.K)
        meas = self.prec_n[bp] > 0
        return self.wsum_n[bp][meas] / self.prec_n[bp][meas]
