"""Where the beds are: exact bed direction, every furrow, and registering each flight to the reference."""
from __future__ import annotations

import json
import math

import numpy as np

from ..paths import DATA
from .io import FIELDS, PROFILE_BIN_M, _polygon, _profile, _sample_pixels, otsu


def estimate_geometry(field, date="05"):
    """Where the beds are. Beds are straight and parallel along the whole field, but NOT evenly spaced across it
    (a bed shaper lays a few beds per tractor pass; on these fields neighbouring beds are 1.6-2.2 m apart). So:
    1) direction: the one that makes the field-wide across-bed profile of plant share sharpest (a 0.1 deg error
       would already smear a bed by ~1 m over 600 m);
    2) the furrows: the deep minima of that profile; a bed is the strip between two neighbouring furrows.
    The profile is kept so that every other flight can be registered to it (see extract)."""
    from scipy.ndimage import uniform_filter1d
    from scipy.signal import find_peaks
    geo, pts = _polygon(FIELDS[field]["polygon"])
    e0, n0 = pts[:, 0].mean(), pts[:, 1].mean()
    E, N, P = _sample_pixels(field, date)
    thr = otsu(P)
    E, N, plant = (E - e0).astype("float32"), (N - n0).astype("float32"), (P > thr).astype("float32")
    half = float(np.hypot(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]))) / 2 + 5
    lo, n_bins = -half, int(2 * half / PROFILE_BIN_M) + 1

    def sharpness(theta_deg, sel):
        th = math.radians(theta_deg)
        vv = -E[sel] * math.sin(th) + N[sel] * math.cos(th)
        pr, n = _profile(None, vv, plant[sel], lo, n_bins)
        ok = n > 5
        return float(np.var(pr[ok]))

    # the sharpness peak is ~(bed width / extent) wide: ~2 deg on a 60-m patch, ~0.1 deg on the whole field.
    # So: coarse on a patch, then finer and finer on more and more of the field.
    r2 = E ** 2 + N ** 2
    patch = np.flatnonzero(r2 < max(30.0, float(np.sqrt(np.percentile(r2, 2)))) ** 2)
    theta = float(np.arange(0, 180, 0.25)[np.argmax([sharpness(t, patch) for t in np.arange(0, 180, 0.25)])])
    for width, stp, sel in ((0.5, 0.01, slice(None, None, 4)), (0.02, 0.0005, slice(None))):
        grid = np.arange(theta - width, theta + width, stp)
        theta = float(grid[np.argmax([sharpness(t, sel) for t in grid])])
    theta %= 180
    th = math.radians(theta)
    v = -E * math.sin(th) + N * math.cos(th)
    pr, n = _profile(None, v, plant, lo, n_bins)
    ok = n > 5
    sm = np.where(ok, uniform_filter1d(np.where(ok, pr, 1.0), int(0.2 / PROFILE_BIN_M)), 1.0)
    fur, _ = find_peaks(-sm, distance=int(1.0 / PROFILE_BIN_M), prominence=0.3)
    furrows = lo + (fur + 0.5) * PROFILE_BIN_M
    gaps = np.diff(furrows)
    med = float(np.median(gaps))
    # a missed furrow (gap > 1.6 x the usual bed): split the gap evenly
    out = [furrows[0]]
    for a, b in zip(furrows[:-1], furrows[1:]):
        k = int(round((b - a) / med)) if (b - a) > 1.6 * med else 1
        out += [a + (b - a) * q / k for q in range(1, k + 1)]
    furrows = np.array(out)
    # the two halves of the field along the beds must show the same profile (straight beds)
    u = E * math.cos(th) + N * math.sin(th)
    h1 = u < np.median(u)
    pa, na = _profile(None, v[h1], plant[h1], lo, n_bins); pb, nb = _profile(None, v[~h1], plant[~h1], lo, n_bins)
    both = (na > 5) & (nb > 5)
    widths = np.diff(furrows)
    return dict(theta_deg=theta, origin=[float(e0), float(n0)], furrows_m=[float(x) for x in furrows],
                bed_spacing_m=float(np.median(widths)), bed_width_p5_p95=[float(np.percentile(widths, 5)), float(np.percentile(widths, 95))],
                beds=int(len(furrows) - 1), halves_agree_r=float(np.corrcoef(pa[both], pb[both])[0, 1]),
                soil_plant_ndvi=float(thr), from_flight=date, profile_lo=float(lo),
                profile=[round(float(x), 3) for x in np.where(ok, pr, -1.0)])


def cmd_geometry(a):
    out = {}
    for f in (a.field or list(FIELDS)):
        g = estimate_geometry(f)
        out[f] = g
        print(f"{f}: beds run {g['theta_deg']:.3f} deg from East; {g['beds']} beds between furrows, "
              f"{g['bed_spacing_m']:.2f} m apart typically ({g['bed_width_p5_p95'][0]:.2f}-{g['bed_width_p5_p95'][1]:.2f} m); "
              f"the two halves of the field agree r = {g['halves_agree_r']:.2f} (straight beds)")
    old = json.loads((DATA / "geometry.json").read_text()) if (DATA / "geometry.json").exists() else {}
    old.update(out)
    DATA.mkdir(exist_ok=True)
    (DATA / "geometry.json").write_text(json.dumps(old, indent=2))
    print("saved", DATA / "geometry.json")


def load_geometry(field):
    return json.loads((DATA / "geometry.json").read_text())[field]


# --------------------------------------------------------------------------------------------- extraction
def register(field, date, g):
    """How far this flight's beds sit across from the reference flight's (orthomosaics of different dates are
    offset by a few decimetres: +0.22 m Jan 31, -0.34 m Mar 18 on Field 1): the shift within half a bed that
    best lines up this flight's across-bed profile with the reference.
    Also returns this flight's soil/plant NDVI split."""
    e0, n0 = g["origin"]
    th = math.radians(g["theta_deg"])
    E, N, P = _sample_pixels(field, date, n_strips=24)
    thr = otsu(P)
    v = -(E - e0) * math.sin(th) + (N - n0) * math.cos(th)
    ref = np.array(g["profile"])
    pr, n = _profile(None, v, (P > thr).astype(float), g["profile_lo"], len(ref))
    ok = (n > 5) & (ref >= 0)
    a = np.where(ok, pr - pr[ok].mean(), 0.0); b = np.where(ok, ref - ref[ok].mean(), 0.0)
    # beds repeat every ~1.9 m, so shifts a whole bed apart match almost equally well; the maps are PPK-georeferenced
    # (dates differ by a few dm), so only shifts under half a bed are physical
    lags = np.arange(-int(0.9 / PROFILE_BIN_M), int(0.9 / PROFILE_BIN_M) + 1)
    cc = np.array([np.sum(np.roll(a, -l) * b) for l in lags]) / np.sqrt(np.sum(a ** 2) * np.sum(b ** 2))
    best = int(np.argmax(cc))
    return float(lags[best] * PROFILE_BIN_M), float(cc[best]), thr
