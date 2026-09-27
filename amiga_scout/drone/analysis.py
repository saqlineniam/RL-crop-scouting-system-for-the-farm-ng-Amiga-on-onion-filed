"""Statistics of the robot-view maps, the yield check at the lab's sample points, and the DEM check."""
from __future__ import annotations

import json
import math
import warnings

import numpy as np

from ..paths import DATA, FIGS
from .extract import load_map
from .io import CRS, DATES, FIELDS, ROOT, STEP_M, _polygon


# --------------------------------------------------------------------------------------------- analysis
def spot_stats(m):
    """Numbers the simulator needs, from one flight (planted cells only)."""
    leaf, cover, mapped = m["leaf_ndvi"], m["cover"], m["mapped"]
    good = (mapped > 0.6) & np.isfinite(leaf)
    med_cover = np.nanmedian(cover[good])
    planted = good & (cover > 0.25 * med_cover)
    x = leaf[planted]
    # spatial structure along the beds: correlation of spot NDVI at a lag of L spots
    dev = np.where(planted, leaf - np.nanmedian(x), np.nan)
    def corr(lag, axis):
        a = dev if axis == 1 else dev
        p = a[:, :-lag] if axis == 1 else a[:-lag, :]
        q = a[:, lag:] if axis == 1 else a[lag:, :]
        ok = np.isfinite(p) & np.isfinite(q)
        return float(np.corrcoef(p[ok], q[ok])[0, 1]) if ok.sum() > 100 else float("nan")
    lags_along = [1, 2, 6, 12, 30, 60]            # spots (0.5 m): 0.5, 1, 3, 6, 15, 30 m
    lags_across = [1, 2, 4, 8, 16]                # beds (1.9 m)
    return dict(
        planted_acres=float((planted * np.diff(m["furrows_m"])[:, None]).sum() * m["step_m"] / 4046.86),
        leaf_median=float(np.median(x)), leaf_sd=float(np.std(x)), leaf_p5=float(np.percentile(x, 5)),
        cover_median=float(med_cover), cover_sd=float(np.nanstd(cover[planted])),
        bare_share=float((good & ~planted).sum() / good.sum()),
        corr_along={l * STEP_M: corr(l, 1) for l in lags_along},
        corr_across={l: corr(l, 0) for l in lags_across},
        r_leaf_cover=float(np.corrcoef(leaf[planted], cover[planted])[0, 1]),
    )


def yield_table():
    """Correlation of the lab's per-point drone values with measured yield. Uses onions_dataset_all.xlsx: the lab's
    plant-pixel band means and canopy area for every sample point and flight, with the measured yield (rows are
    matched by field and sample id - the sheet is not in id order)."""
    import pandas as pd
    a = pd.read_excel(f"{ROOT}/Dataset/onions_dataset_all.xlsx", sheet_name="dataset")
    rows = []
    for f, fld in (("f1", 1), ("f2", 2)):
        A = a[a.Field == fld].set_index("id").sort_index()
        y = A.Yield.values
        for i, d in enumerate(DATES, 1):
            ndvi = ((A[f"NIR_{i}"] - A[f"Red_{i}"]) / (A[f"NIR_{i}"] + A[f"Red_{i}"])).values
            rows.append((f, d, float(np.corrcoef(ndvi, y)[0, 1]), float(np.corrcoef(A[f"area_{i}"].values, y)[0, 1])))
    return rows


def dem_check():
    """Median 'canopy height' (DEM minus the local furrow bottom) per flight - to show the DEM can't give plant height."""
    import rasterio
    from rasterio.warp import reproject, Resampling
    from rasterio.transform import from_origin
    from rasterio.features import geometry_mask
    from scipy.ndimage import minimum_filter, uniform_filter
    res, out = 0.25, {}
    for f, cfg in FIELDS.items():
        geo, pts = _polygon(cfg["polygon"])
        W, H = int(np.ptp(pts[:, 0]) / res) + 1, int(np.ptp(pts[:, 1]) / res) + 1
        tr = from_origin(pts[:, 0].min(), pts[:, 1].max(), res, res)
        core = minimum_filter((~geometry_mask(geo, out_shape=(H, W), transform=tr)).astype(np.uint8), size=int(15 / res)).astype(bool)
        for d, c in cfg["dem_chunks"].items():
            with rasterio.open(f"{ROOT}/Project/Metashape/Onions.files/{c}/0/elevation/tile-0-0.tif") as r:
                dem = np.full((H, W), np.nan, "float32")
                reproject(r.read(1), dem, src_transform=r.transform, src_crs=r.crs, src_nodata=r.nodata,
                          dst_transform=tr, dst_crs=CRS, dst_nodata=np.nan, resampling=Resampling.bilinear)
                px_m = abs(r.transform.a) * 111320 * math.cos(math.radians(32))
            z = np.where(np.isfinite(dem), dem, np.nanmedian(dem))
            chm = z - uniform_filter(minimum_filter(z, size=int(2.5 / res)), size=int(2.5 / res))
            out[(f, d)] = (px_m, float(np.median(uniform_filter(chm, 3)[core & np.isfinite(dem)])))
    return out


def cmd_analyse(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIGS.mkdir(exist_ok=True)
    lines = []
    say = lambda s="": (print(s), lines.append(s))  # noqa: E731
    have = sorted(p.stem for p in DATA.glob("f?_??.npz"))
    say("DRONE -> ROBOT SPOT STATISTICS (bed-aligned 0.508-m spots, plant pixels only)")
    say(f"{'map':7s} {'date':10s} {'acres':>5s} {'leafNDVI':>8s} {'SD':>5s} {'p5':>5s} {'cover':>5s} {'bare%':>5s} "
        f"{'r(0.5m)':>7s} {'r(3m)':>6s} {'r(15m)':>6s} {'r(1bed)':>7s} {'r(8bed)':>7s} {'r(leaf,cover)':>13s}")
    stats = {}
    for name in have:
        f, d = name.split("_")
        m = load_map(f, d)
        st = stats[name] = spot_stats(m)
        ca, cx = st["corr_along"], st["corr_across"]
        say(f"{name:7s} {DATES[d]:10s} {st['planted_acres']:5.1f} {st['leaf_median']:8.3f} {st['leaf_sd']:5.3f} "
            f"{st['leaf_p5']:5.3f} {st['cover_median']:5.2f} {100 * st['bare_share']:5.1f} {ca[STEP_M]:7.2f} "
            f"{ca[6 * STEP_M]:6.2f} {ca[30 * STEP_M]:6.2f} {cx[1]:7.2f} {cx[8]:7.2f} {st['r_leaf_cover']:13.2f}")
    (DATA / "spot_stats.json").write_text(json.dumps({k: {kk: (vv if not isinstance(vv, dict) else {str(a_): b for a_, b in vv.items()})
                                                          for kk, vv in v.items()} for k, v in stats.items()}, indent=1))
    say()
    say("YIELD CHECK (lab's plant-pixel values at the 50 sample points per field): correlation with measured yield")
    say(f"{'field':5s} {'date':10s} {'NDVI':>6s} {'canopy area':>12s}")
    for f, d, rn, ra in yield_table():
        say(f"{f:5s} {DATES[d]:10s} {rn:6.2f} {ra:12.2f}")
    if a.dem:
        say()
        say("DEM CHECK: median 'canopy height' = DEM - local furrow bottom (real onions: ~20-55 cm over this season)")
        for (f, d), (px_m, h) in dem_check().items():
            say(f"{f} {DATES[d]}: DEM pixel {px_m:.2f} m -> {100 * h:.1f} cm")
    (DATA / "analysis.txt").write_text("\n".join(lines))
    # figure: robot-view maps (leaf NDVI and cover, 10-ft block means) of every extracted flight
    if have:
        fig, axs = plt.subplots(2, len(have), figsize=(2.6 * len(have), 7), squeeze=False)
        for i, name in enumerate(have):
            f, d = name.split("_")
            m = load_map(f, d)
            for row, key, cmap, lab in ((0, "leaf_ndvi", "RdYlGn", "leaf NDVI"), (1, "cover", "YlGn", "canopy cover")):
                z = np.where(m["mapped"] > 0.6, m[key], np.nan)
                nJ = z.shape[1] // 6 * 6
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    blk = np.nanmean(z[:, :nJ].reshape(z.shape[0], -1, 6), axis=2)
                lo, hi = np.nanpercentile(blk, [2, 98])
                im = axs[row, i].imshow(blk, cmap=cmap, vmin=lo, vmax=hi, aspect="auto", interpolation="nearest")
                axs[row, i].set_title(f"{name} {lab}", fontsize=8); axs[row, i].axis("off")
                fig.colorbar(im, ax=axs[row, i], fraction=0.05)
        fig.suptitle("Robot view: one row per bed, one pixel per 10-ft block", fontsize=9)
        fig.tight_layout(); fig.savefig(FIGS / "robot_view_maps.png", dpi=90)
        print("saved", FIGS / "robot_view_maps.png")
    print("saved", DATA / "analysis.txt")
