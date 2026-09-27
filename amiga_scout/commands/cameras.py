"""`cameras`: the camera model for the current settings - the datasheet facts, what they mean at this mounting
height, the errors of one shot at every growth stage, and a figure of the curves (figures/camera_curves.png)."""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from ..config import FLIGHT_DATES, LEAF_WIDTH_MM, STAGE_HEIGHT_CM
from ..evaluation import build_cfg
from ..paths import FIGS
from ..sensors import (D455_DEPTH, D455_RGB, JAI_AD130GE, defocus_px, depth_gsd, depth_of_field, depth_quality,
                       depth_rms, lens_fov_deg, lens_gsd, min_z, shot_statistics)

RESOLUTIONS = ["1280x720", "848x480", "640x360"]
# reference palette, first three categorical slots (validated all-pairs, light surface); text wears text tokens
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def _stage_rows(cfg, rng, n):
    rows = []
    for d, h in STAGE_HEIGHT_CM.items():
        c = replace(cfg, height_mean_cm=h, leaf_width_mm=LEAF_WIDTH_MM[d])
        st = shot_statistics(c, rng, h, LEAF_WIDTH_MM[d], n)
        nodepth = {r: shot_statistics(replace(c, depth_resolution=r), rng, h, LEAF_WIDTH_MM[d], n // 3)["quick"]["no_depth"]
                   for r in RESOLUTIONS}
        rows.append((d, h, st, nodepth))
    return rows


def cmd_cameras(args):
    cfg = build_cfg(args, field="sq5")
    rng = np.random.default_rng(args.seed)
    res, f_nir = cfg.depth_resolution, cfg.nir_lens_focal_mm / 1e3
    print("CAMERAS (datasheet facts; sources in docs/references.md)")
    print(f"  depth  {D455_DEPTH.name}: baseline {D455_DEPTH.baseline_m * 1000:.0f} mm, field of view "
          f"{D455_DEPTH.fov_hd_deg[0]:.0f}x{D455_DEPTH.fov_hd_deg[1]:.0f} deg (16:9), {D455_DEPTH.shutter} shutter\n"
          f"         spec: {D455_DEPTH.accuracy_spec}\n         source: {D455_DEPTH.source}")
    print(f"  RGB    {D455_RGB.name}: {D455_RGB.width_px}x{D455_RGB.height_px}, {D455_RGB.fov_deg[0]:.0f}x"
          f"{D455_RGB.fov_deg[1]:.0f} deg, {D455_RGB.shutter} shutter (YOLO)")
    print(f"  NIR    {JAI_AD130GE.name}: {JAI_AD130GE.width_px}x{JAI_AD130GE.height_px} px of "
          f"{JAI_AD130GE.pixel_m * 1e6:.2f} um, S/N {JAI_AD130GE.snr_db['nir']:.0f} dB (NIR), {JAI_AD130GE.shutter}\n"
          f"         {JAI_AD130GE.notes}\n         source: {JAI_AD130GE.source}")
    zmin = min_z(D455_DEPTH, res)
    near, far = depth_of_field(f_nir, cfg.nir_lens_fnumber, cfg.nir_focus_m, 2 * JAI_AD130GE.pixel_m)
    fov = lens_fov_deg(JAI_AD130GE.sensor_mm, f_nir)
    print(f"\nSETUP (config.py)\n  cameras {cfg.camera_height_m:.2f} m above the ground, looking down")
    print(f"  depth at {res}: no depth closer than {zmin:.2f} m -> plants up to "
          f"{100 * (cfg.camera_height_m - zmin):.0f} cm tall can be measured; at 0.6 m a depth pixel covers "
          f"{1000 * depth_gsd(D455_DEPTH, res, 0.6):.1f} mm and its noise is {1000 * depth_rms(D455_DEPTH, res, 0.6, cfg.depth_subpixel_rms):.1f} mm")
    for r in RESOLUTIONS:
        print(f"      {r:>8s}: Min-Z {min_z(D455_DEPTH, r):.2f} m -> tallest measurable plant "
              f"{100 * (cfg.camera_height_m - min_z(D455_DEPTH, r)):.0f} cm")
    print(f"  NIR: {cfg.nir_lens_focal_mm:g} mm lens at f/{cfg.nir_lens_fnumber:g}, focused at {cfg.nir_focus_m:.2f} m -> "
          f"{fov[0]:.0f}x{fov[1]:.0f} deg; a pixel covers {1000 * lens_gsd(JAI_AD130GE.pixel_m, f_nir, 0.6):.2f} mm at 0.6 m; "
          f"sharp (blur under 2 px) from {near:.2f} to {far:.2f} m")
    rows = _stage_rows(cfg, rng, args.shots)
    print(f"\nONE SHOT, BY GROWTH STAGE ({args.shots} simulated shots each; quick = one frame where the robot stopped,"
          f" careful = from the best spot seen on the way in, {cfg.careful_average_s:g}-s average, 2nd NIR shot if needed)")
    print(f"  {'date':7s} {'height':>6s} {'leaf':>5s} {'to top':>6s} | {'depth q':>7s} {'no depth':>8s} "
          f"{'height error quick':>19s} {'careful':>14s} | {'NIR q':>5s} {'NDVI error quick':>17s} {'careful':>15s}")
    for d, h, st, _ in rows:
        q, c = st["quick"], st["careful"]
        print(f"  {FLIGHT_DATES[d]:7s} {h:4.0f}cm {LEAF_WIDTH_MM[d]:3.0f}mm {cfg.camera_height_m - h / 100:5.2f}m | "
              f"{q['depth_quality']:7.2f} {100 * q['no_depth']:7.1f}% {q['height_bias']:+6.2f} +- {q['height_sd']:4.2f} cm "
              f"{c['height_bias']:+5.2f} +- {c['height_sd']:4.2f} | {q['nir_quality']:5.2f} {q['ndvi_bias']:+7.3f} +- "
              f"{q['ndvi_sd']:5.3f} {c['ndvi_bias']:+7.3f} +- {c['ndvi_sd']:5.3f}")
    print("  (bias = mean error, +- = its spread; the NDVI calibration offset of a mission is not included)")
    print("  shots without depth, by depth resolution: " + "; ".join(
        f"{FLIGHT_DATES[d]}: " + "/".join(f"{100 * nd[r]:.0f}%" for r in RESOLUTIONS) for d, _, _, nd in rows)
          + f"   ({' / '.join(RESOLUTIONS)})")
    if not args.no_figure:
        _figure(cfg, rows, args.out or FIGS / "camera_curves.png")


def _style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.set_xlabel(xlabel, fontsize=9, color=INK_2)
    ax.set_ylabel(ylabel, fontsize=9, color=INK_2)
    ax.grid(True, color=GRID, linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8)


def _line(ax, x, y, k, label, marker=False):
    """One series: a 2-px line (dots with a surface ring for per-stage values). Returns its end for labelling."""
    ax.plot(x, y, color=SERIES[k], linewidth=2, solid_capstyle="round", label=label,
            **(dict(marker="o", markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5) if marker else {}))
    j = int(np.nanargmax(np.where(np.isfinite(y), np.arange(len(y)), -1)))
    return float(x[j]), float(y[j]), label


def _end_labels(ax, ends):
    """Direct labels just right of the line ends, pushed apart vertically so they never overlap (text ink)."""
    lo, hi = ax.get_ylim()
    gap = 0.05 * (hi - lo)
    x0, x1 = ax.get_xlim()
    placed = []
    for x, y, label in sorted(ends, key=lambda e: e[1]):
        y_text = max(y, placed[-1] + gap) if placed else y
        placed.append(y_text)
        ax.text(x + 0.02 * (x1 - x0), y_text, label, fontsize=7.5, color=INK_2, va="center")
    ax.set_xlim(x0, x1 + 0.2 * (x1 - x0))


def _figure(cfg, rows, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 3, figsize=(16, 9), facecolor=SURFACE)
    z = np.linspace(0.15, 1.3, 300)
    tops = (cfg.camera_height_m - max(STAGE_HEIGHT_CM.values()) / 100,
            cfg.camera_height_m - min(STAGE_HEIGHT_CM.values()) / 100)

    ax = axs[0, 0]
    ends = []
    for k, r in enumerate(RESOLUTIONS):
        y = np.array([1000 * depth_rms(D455_DEPTH, r, v, cfg.depth_subpixel_rms) if v >= min_z(D455_DEPTH, r) else np.nan
                      for v in z])
        ends.append(_line(ax, z, y, k, r))
    ax.axhline(cfg.depth_systematic_mm, color=INK_2, linewidth=0.8)
    ax.text(0.16, cfg.depth_systematic_mm + 0.1, "systematic error (Servi et al. 2021)", fontsize=7.5, color=INK_2)
    ax.axvspan(*tops, color=GRID, alpha=0.6, linewidth=0)
    _style(ax, "D455 depth noise of one pixel", "camera to plant top (m)   grey: onion tops, Jan-Apr",
           "noise (mm); each line starts at its Min-Z")
    ax.set_ylim(0, cfg.depth_systematic_mm * 1.25)
    _end_labels(ax, ends)
    ax.legend(fontsize=7.5, frameon=False, title="depth resolution", title_fontsize=7.5, loc="center left")

    ax = axs[0, 1]
    ends = [_line(ax, z, np.array([depth_quality(cfg, v, 0.0, 0.0, w / 1e3) for v in z]), k, f"leaf {w} mm")
            for k, w in enumerate((5, 10, 15))]
    ax.axvspan(*tops, color=GRID, alpha=0.6, linewidth=0)
    _style(ax, f"D455 depth quality on a plant ({cfg.depth_resolution})", "camera to plant top (m)",
           "share of plant pixels with depth")
    ax.set_ylim(0, 1)
    _end_labels(ax, ends)
    ax.legend(fontsize=7.5, frameon=False, title="onion leaf width", title_fontsize=7.5, loc="upper right")

    ax = axs[0, 2]
    f, px = cfg.nir_lens_focal_mm / 1e3, JAI_AD130GE.pixel_m
    gsd = np.array([1000 * lens_gsd(px, f, v) for v in z])
    focus = np.array([1000 * defocus_px(f, cfg.nir_lens_fnumber, cfg.nir_focus_m, v, px) * lens_gsd(px, f, v) for v in z])
    motion = np.full_like(z, 1000 * cfg.leaf_sway_speed_mps * 3 * cfg.nir_exposure_ms / 1e3)
    ends = [_line(ax, z, gsd, 0, "pixel size"), _line(ax, z, focus, 1, "focus blur"),
            _line(ax, z, motion, 2, "leaf motion blur (windy)")]
    _style(ax, f"NIR camera detail ({cfg.nir_lens_focal_mm:g} mm lens, f/{cfg.nir_lens_fnumber:g}, "
           f"focused at {cfg.nir_focus_m:g} m)", "camera to mid-canopy (m)", "size on the plant (mm)")
    ax.set_ylim(0, None)
    _end_labels(ax, ends)
    ax.legend(fontsize=7.5, frameon=False, loc="upper center")

    labels = [FLIGHT_DATES[d] for d, *_ in rows]
    x = np.arange(len(rows), dtype=float)
    for ax, key, unit, title in ((axs[1, 0], "height_sd", "cm", "Height error of one plant, one visit"),
                                 (axs[1, 1], "ndvi_sd", "NDVI", "NDVI error of one plant, one visit")):
        ends = [_line(ax, x, np.array([st[mode][key] for _, _, st, _ in rows]), k, mode, marker=True)
                for k, mode in enumerate(("quick", "careful"))]
        _style(ax, title, "flight date (growth stage)", f"error spread ({unit})")
        ax.set_xticks(x, labels)
        ax.set_ylim(0, None)
        _end_labels(ax, ends)
        ax.legend(fontsize=7.5, frameon=False, loc="lower left")

    ax = axs[1, 2]
    ends = [_line(ax, x, np.array([100 * nd[r] for *_, nd in rows]), k, r, marker=True)
            for k, r in enumerate(RESOLUTIONS)]
    _style(ax, f"Plants with no height at all (cameras {cfg.camera_height_m:g} m up)", "flight date (growth stage)",
           "shots without depth (%)")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 100)
    _end_labels(ax, ends)
    ax.legend(fontsize=7.5, frameon=False, title="depth resolution", title_fontsize=7.5, loc="upper left")

    fig.suptitle("Camera model for the current settings: Intel RealSense D455 (height) and JAI AD-130GE (NDVI)",
                 fontsize=12, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    FIGS.mkdir(exist_ok=True)
    fig.savefig(out, dpi=110, facecolor=SURFACE)
    print(f"\nsaved {out}")
