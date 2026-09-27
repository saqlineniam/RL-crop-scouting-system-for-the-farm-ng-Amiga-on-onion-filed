"""Streaming an orthomosaic into the bed-aligned robot-view map (one cell per bed and 0.508-m spot)."""
from __future__ import annotations

import json
import math
import os

import numpy as np

from ..paths import DATA
from .beds import load_geometry, register
from .io import DATES, FIELDS, STEP_M, _ndvi, _polygon, ortho_path


def extract(field, date, strip=512):
    """Stream one orthomosaic and reduce it to the bed-aligned spot grid: one cell per real bed (the strip between
    two neighbouring furrows) and per 0.508 m along it. The flight is first registered to the reference flight."""
    import rasterio
    from rasterio.windows import Window
    from rasterio.features import geometry_mask
    g = load_geometry(field)
    geo, pts = _polygon(FIELDS[field]["polygon"])
    th = math.radians(g["theta_deg"])
    e0, n0 = g["origin"]
    ct, st = math.cos(th), math.sin(th)
    shift, match, thr = register(field, date, g)
    furrows = np.array(g["furrows_m"]) + shift
    widths = np.diff(furrows)
    pu = (pts[:, 0] - e0) * ct + (pts[:, 1] - n0) * st
    umin = float(pu.min())
    nK, nJ = len(furrows) - 1, int(math.ceil((pu.max() - umin) / STEP_M))
    size = nK * nJ
    acc = {k: np.zeros(size) for k in ("n_all", "n_plant", "s_plant", "s_all")}
    with rasterio.open(ortho_path(field, date)) as r:
        px = r.res[0]
        # only the columns/rows of the field's bounding box
        c0 = max(0, int((pts[:, 0].min() - r.transform.c) / px) - 2); c1 = min(r.width, int((pts[:, 0].max() - r.transform.c) / px) + 2)
        r0 = max(0, int((r.transform.f - pts[:, 1].max()) / px) - 2); r1 = min(r.height, int((r.transform.f - pts[:, 1].min()) / px) + 2)
        E = r.transform.c + px * (np.arange(c0, c1) + 0.5) - e0
        for y0 in range(r0, r1, strip):
            h = min(strip, r1 - y0)
            w = Window(c0, y0, c1 - c0, h)
            red, nir = r.read([2, 4], window=w).astype("float32")
            nd = _ndvi(red, nir)
            inside = ~geometry_mask(geo, out_shape=nd.shape, transform=r.window_transform(w))
            ok = np.isfinite(nd) & inside
            if not ok.any():
                continue
            N = (r.transform.f - px * (np.arange(y0, y0 + h) + 0.5) - n0)[:, None]
            u = (E[None, :] * ct + N * st)[ok]
            v = (-E[None, :] * st + N * ct)[ok]
            k = np.searchsorted(furrows, v) - 1                  # the bed between furrow k and furrow k+1
            j = np.floor((u - umin) / STEP_M).astype(np.int64)
            keep = (k >= 0) & (k < nK) & (j >= 0) & (j < nJ)
            key = (k * nJ + j)[keep]
            ndv = nd[ok][keep]
            plant = ndv > thr
            acc["n_all"] += np.bincount(key, minlength=size)
            acc["s_all"] += np.bincount(key, weights=ndv, minlength=size)
            acc["n_plant"] += np.bincount(key[plant], minlength=size)
            acc["s_plant"] += np.bincount(key[plant], weights=ndv[plant], minlength=size)
            print(f"\r  {field} {date}: {100 * (y0 + h - r0) // (r1 - r0)}%", end="", flush=True)
        print()
    expected = (STEP_M * widths / px ** 2)[:, None]               # pixels of a whole cell (beds differ in width)
    n_all = acc["n_all"].reshape(nK, nJ)
    with np.errstate(invalid="ignore", divide="ignore"):
        leaf = (acc["s_plant"] / acc["n_plant"]).reshape(nK, nJ)
        cover = (acc["n_plant"] / acc["n_all"]).reshape(nK, nJ)
        mean_ndvi = (acc["s_all"] / acc["n_all"]).reshape(nK, nJ)
    mapped = n_all / expected                         # share of the cell that is inside the field and mapped
    out = DATA / f"{field}_{date}.npz"
    np.savez_compressed(out, leaf_ndvi=leaf.astype("float32"), cover=cover.astype("float32"),
                        mean_ndvi=mean_ndvi.astype("float32"), mapped=mapped.astype("float32"),
                        furrows_m=furrows, umin=umin, step_m=STEP_M, soil_plant_ndvi=thr, field=field, date=date,
                        shift_m=shift, registration_r=match,
                        geometry=json.dumps({k: v for k, v in g.items() if k != "profile"}))
    good = mapped > 0.6
    print(f"  saved {out.name}: {nK} beds x {nJ} spots, {(good * widths[:, None]).sum() * STEP_M / 4046.86:.1f} acres mapped; "
          f"registered to the {g['from_flight']} flight: shifted {shift:+.2f} m across the beds (profile match r = {match:.2f}); "
          f"soil/plant NDVI split {thr:.3f}")
    return out


def cmd_extract(a):
    extract(a.field, a.date)


def cmd_extract_all(a):
    for f in FIELDS:
        for d in DATES:
            if (DATA / f"{f}_{d}.npz").exists() and not a.redo:
                print("have", f"{f}_{d}.npz"); continue
            if not os.path.exists(ortho_path(f, d)):
                print("missing", ortho_path(f, d)); continue
            extract(f, d)


def load_map(field, date):
    m = np.load(DATA / f"{field}_{date}.npz")
    return {k: m[k] for k in m.files}
