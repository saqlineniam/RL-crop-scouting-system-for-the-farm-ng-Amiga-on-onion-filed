"""`mount`: which fixed camera height (and depth resolution, and NIR lens) works for the whole season?

For every height and depth resolution it checks, using the real plant heights of the onion maps (all flights):
    too close     share of plants whose top is closer than the D455's minimum distance (no height at all)
    rows covered  the depth and RGB cameras see the whole canopy across the bed (BED_CANOPY_WIDTH_M) at the tallest
                  plants; for the NIR camera it picks the longest standard lens that still covers it (a longer lens =
                  more detail), focuses it at the season's typical distance and stops down until the season is sharp
    errors        per-shot height and NDVI errors at every growth stage (the camera model, positioned as the
                  hand-written rules do): root-mean-square, so a steady offset (e.g. thin seedling leaves mixed
                  with soil in coarse pixels -> NDVI low) counts as well as the scatter
and recommends the height with the smallest errors among those that satisfy the first two.
"""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from ..config import BED_CANOPY_WIDTH_M, FLIGHT_DATES, LEAF_WIDTH_MM, SCENARIOS, STAGE_HEIGHT_CM, Cfg
from ..env import AmigaMissionEnv
from ..evaluation import parse_sets
from ..fields import available_maps
from ..sensors import (D455_DEPTH, JAI_AD130GE, depth_footprint, depth_of_field, min_z, rgb_footprint,
                       shot_statistics)

LENSES_MM = (2.8, 3.5, 4.0, 4.5, 5.0, 6.0, 8.0, 12.0)     # common C-mount focal lengths
FNUMBERS = (2.0, 2.8, 4.0, 5.6, 8.0)
RESOLUTIONS = ("1280x720", "848x480", "640x360")


def plant_heights():
    """Plant heights (cm) the simulator's truth gives on every extracted onion map, by flight date."""
    out = {}
    for name in available_maps():
        env = AmigaMissionEnv(replace(Cfg(), field=name, route_start_bed=0, route_corner=0, **SCENARIOS["as_is"]))
        out.setdefault(name.split("_")[1], []).append(env.height[env.vplant])
    return {d: np.concatenate(v) for d, v in out.items()}


def nir_setup(z_cover, z_mid):
    """(lens mm or None, focus m, f-number) for the NIR camera: the longest standard lens whose view across the bed
    still covers the canopy at the tallest plants; focused at the season's median mid-canopy distance; the widest
    aperture that keeps blur under 2 pixels from the season's 1st to 99th percentile distance."""
    fits = [f for f in LENSES_MM if JAI_AD130GE.sensor_mm[0] * z_cover / f >= BED_CANOPY_WIDTH_M]
    if not fits:
        return None, None, None
    f = max(fits)
    focus = float(np.median(z_mid))
    lo, hi = np.percentile(z_mid, [1, 99])
    for n in FNUMBERS:
        near, far = depth_of_field(f / 1e3, n, focus, 2 * JAI_AD130GE.pixel_m)
        if near <= lo and far >= hi:
            return f, focus, n
    return f, focus, FNUMBERS[-1]


def cmd_mount(args):
    base = replace(Cfg(), **parse_sets(args.set))
    heights = plant_heights()
    if not heights:
        raise SystemExit("no onion maps: run `python -m amiga_scout extract-all` first")
    pooled = np.concatenate(list(heights.values()))
    tallest = float(np.percentile(pooled, 99))
    rng = np.random.default_rng(args.seed)
    print(f"plants on the maps: {len(pooled):,}; tallest (99th percentile) {tallest:.0f} cm; canopy to cover across the "
          f"bed {BED_CANOPY_WIDTH_M:.2f} m")
    print("  by flight: " + ", ".join(f"{FLIGHT_DATES[d]} {np.median(h):.0f} cm (99%: {np.percentile(h, 99):.0f})"
                                      for d, h in sorted(heights.items())))
    rows = []
    for H in np.arange(args.min, args.max + 1e-9, args.step):
        H = round(float(H), 3)
        z_cover = H - tallest / 100
        z_mid = H - pooled / 200
        f, focus, n = nir_setup(z_cover, z_mid) if z_cover > 0 else (None, None, None)
        for res in RESOLUTIONS:
            too_close = float(np.mean(H - pooled / 100 < min_z(D455_DEPTH, res)))
            depth_cover = depth_footprint(replace(base, depth_resolution=res), max(z_cover, 0.0))[1]
            rgb_cover = rgb_footprint(max(z_cover, 0.0))[1]
            ok = (too_close <= args.max_too_close and depth_cover >= BED_CANOPY_WIDTH_M
                  and rgb_cover >= BED_CANOPY_WIDTH_M and f is not None)
            errs = []
            if f is not None:
                for d, h in STAGE_HEIGHT_CM.items():
                    c = replace(base, camera_height_m=H, depth_resolution=res, nir_lens_focal_mm=f, nir_focus_m=focus,
                                nir_lens_fnumber=n, height_mean_cm=h, leaf_width_mm=LEAF_WIDTH_MM[d])
                    errs.append(shot_statistics(c, rng, h, LEAF_WIDTH_MM[d], args.shots)["careful"])
            if errs:
                h_sd = float(np.nanmean([math.hypot(e["height_sd"], e["height_bias"]) for e in errs]))
                n_sd = float(np.nanmean([math.hypot(e["ndvi_sd"], e["ndvi_bias"]) for e in errs]))
                n_worst = float(np.nanmax([math.hypot(e["ndvi_sd"], e["ndvi_bias"]) for e in errs]))
                no_h = float(np.mean([e["no_depth"] for e in errs]))
                score = h_sd / 1.0 + n_sd / 0.02 + 10 * no_h          # ~1 cm and ~0.02 NDVI weigh the same
            else:
                h_sd = n_sd = n_worst = no_h = score = math.nan
            rows.append(dict(H=H, res=res, too_close=too_close, depth_cover=depth_cover, rgb_cover=rgb_cover, lens=f,
                             focus=focus, fnum=n, h_sd=h_sd, n_sd=n_sd, n_worst=n_worst, no_h=no_h, score=score, ok=ok))
    print(f"\n{'height':>6s} {'depth res':>9s} {'too close':>9s} {'depth/RGB view across':>22s} {'NIR lens':>15s} "
          f"{'height err':>10s} {'NDVI err':>8s} {'worst':>6s} {'no height':>9s}  verdict")
    for r in rows:
        if abs(r["H"] * 10 - round(r["H"] * 10)) > 1e-6 and not args.all:
            continue
        lens = f"{r['lens']:g} mm f/{r['fnum']:g}" if r["lens"] else "none covers"
        verdict = "OK" if r["ok"] else ("too close" if r["too_close"] > args.max_too_close else "rows not covered")
        print(f"{r['H']:5.2f}m {r['res']:>9s} {100 * r['too_close']:8.1f}% {r['depth_cover']:9.2f} / {r['rgb_cover']:4.2f} m "
              f"{lens:>15s} {r['h_sd']:8.2f}cm {r['n_sd']:8.3f} {r['n_worst']:6.3f} {100 * r['no_h']:8.1f}%  {verdict}")
    good = [r for r in rows if r["ok"] and np.isfinite(r["score"])]
    if not good:
        print("\nno height satisfies both requirements in this range - widen --min/--max")
        return
    best = min(good, key=lambda r: r["score"])
    print(f"\nRECOMMENDED: cameras {best['H']:.2f} m above the ground, D455 depth at {best['res']}, NIR camera with a "
          f"{best['lens']:g} mm lens at f/{best['fnum']:g} focused at {best['focus']:.2f} m")
    print(f"  no plant closer than the D455's minimum distance ({100 * best['too_close']:.1f}% of plants), the depth and "
          f"RGB views cover {best['depth_cover']:.2f} / {best['rgb_cover']:.2f} m across the bed at the tallest plants")
    print(f"  season-average error (RMS) of one careful plant: height {best['h_sd']:.2f} cm, NDVI {best['n_sd']:.3f} "
          f"(worst growth stage {best['n_worst']:.3f}); "
          f"{100 * best['no_h']:.1f}% of plants without height (seedlings too thin to resolve)")
    print(f"  set it with:  --set camera_height_m={best['H']:g} --set depth_resolution={best['res']} "
          f"--set nir_lens_focal_mm={best['lens']:g} --set nir_lens_fnumber={best['fnum']:g} "
          f"--set nir_focus_m={best['focus']:.2f}   (or edit config.py)")
