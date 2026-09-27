"""The fields a mission can run on: real maps (robot-view maps of the onion survey, see amiga_scout.drone), their
TRAINING and TEST regions (a spatial 80/20 split of each field, see split_regions), random PIECES of them, and
synthetic square fields ("sq10"); pools of them ("f1~train f2~train@10") are drawn from per mission.

    f1_05            Field 1, flight 05 (Apr 4), whole
    f1_05~test       its TEST region: one compact end with 20% of the crop (never trained on)
    f1_05~train      its TRAINING region: the rest, minus a buffer strip next to the test region
    f1_05~train@10   a random 10-acre piece of the training region (a new piece every mission)
    f1_05@10         a random 10-acre piece anywhere in the field
    f1_05~train*2    the training region laid out twice side by side (the copy mirrored, so the texture runs on
                     across the seam): a BIG field made only of real drone data from that region - f1_05~train*2@30
                     is a 30-acre piece of it. This is how the robot trains on 25-40-acre fields without ever seeing
                     the test regions; f1~test*5 is a 30-acre field made only of the test region (a held-out big
                     field). *3x2 = 3 copies across the beds x 2 along them
    f1, f1~train...  the same for every extracted flight of Field 1
"""
from __future__ import annotations

import json
import math

import numpy as np

from .config import BARE_COVER_FRAC, SPLIT_BUFFER_M, SPLIT_TEST_SHARE
from .paths import DATA

REGIONS = ("train", "test")


def square_field(acres):
    """Settings for a synthetic square field of this size."""
    return dict(field=f"sq{acres:g}", field_acres=float(acres), row_length_m=math.sqrt(acres * 4046.86))


# ---- names ----------------------------------------------------------------------------------------------
def available_maps():
    return sorted(p.stem for p in DATA.glob("f?_??.npz"))


def parse_name(name):
    """'f1_05~train*2@10' -> ('f1_05', 'train', 10.0); 'f1_05' -> ('f1_05', None, None) (the tiling: tile_of)."""
    base, _, size = str(name).partition("@")
    base = base.partition("*")[0]
    base, _, region = base.partition("~")
    return base, region or None, (float(size) if size else None)


def tile_of(name):
    """'f1_05~train*2@30' -> (2, 1): copies of the region side by side across the beds, and along them
    ('*3x2' -> (3, 2)); (1, 1) = not tiled."""
    t = str(name).partition("@")[0].partition("*")[2]
    if not t:
        return 1, 1
    a, _, b = t.partition("x")
    return int(a), int(b or 1)


def base_map(name):
    """'f1_05~train@10' -> 'f1_05' (the map a region or piece is cut from)."""
    return parse_name(name)[0]


def region_of(name):
    """'f1_05~test' -> 'test'; None for a whole field or a piece cut anywhere in it."""
    return parse_name(name)[1]


def field_key(name):
    """'f1_05~test' -> 'f1' (the field, all flights)."""
    return base_map(name).split("_")[0]


def field_pool(spec):
    """Expand a field spec into the list of fields a mission is drawn from (space-separated tokens):
        f1_05       Field 1, flight 05, whole            f1           every extracted Field 1 flight, whole
        f1_05~test  its test region (~train: training)   f1~train@10  10-acre pieces of Field 1's training region
        f1_05@10    a random 10-acre piece of f1_05      sq10         a synthetic 10-acre square field"""
    out = []
    for tok in str(spec).split():
        if tok.startswith("sq"):
            float(tok[2:])
            out.append(tok)
            continue
        base, at, size = tok.partition("@")
        base, star, tile = base.partition("*")
        base, tilde, region = base.partition("~")
        if at and not float(size) > 0:
            raise SystemExit(f"bad piece size in {tok!r}")
        if star and not all(p.isdigit() and int(p) >= 1 for p in tile.split("x")) or tile.count("x") > 1:
            raise SystemExit(f"bad tiling in {tok!r}: use *2 (copies across the beds) or *3x2 (across x along)")
        if tilde and region not in REGIONS:
            raise SystemExit(f"bad region in {tok!r}: use ~train or ~test")
        if "_" in base:
            if not (DATA / f"{base}.npz").exists():
                raise SystemExit(f"no map {base}: run `python -m amiga_scout extract --field "
                                 f"{base[:2]} --date {base[3:]}` first")
            maps = [base]
        else:
            maps = [m for m in available_maps() if m.startswith(base + "_")]
            if not maps:
                raise SystemExit(f"no maps for {base}: run `python -m amiga_scout extract-all` first")
        out += [m + tilde + region + star + tile + at + size for m in maps]
    if not out:
        raise SystemExit(f"empty field spec {spec!r}")
    return out


def overlaps(a, b):
    """Can missions on fields a and b see the same crop? (the same field, and not two different regions of it)"""
    ra, rb = region_of(a), region_of(b)
    if str(a).startswith("sq") or str(b).startswith("sq") or field_key(a) != field_key(b):
        return False
    return ra is None or rb is None or ra == rb


# ---- real fields --------------------------------------------------------------------------------------
_MAPS = {}
_SPLITS = {}


def load_field(name):
    """A real field's robot-view map (amiga_scout.drone), prepared once per process: plant-pixel NDVI and canopy
    cover per bed x 0.508-m spot, which spots are crop, and the field's medians and size. For a region or piece
    ("f1_05~train@10") this is the whole map it is cut from - the medians stay the whole field's."""
    name = base_map(name)
    if name not in _MAPS:
        m = np.load(DATA / f"{name}.npz")
        leaf, cover, mapped = m["leaf_ndvi"], m["cover"], m["mapped"]
        geo = json.loads(str(m["geometry"]))
        good = (mapped > 0.6) & np.isfinite(cover)
        med_cover = float(np.nanmedian(cover[good]))
        planted = good & (cover >= BARE_COVER_FRAC * med_cover) & np.isfinite(leaf)
        bed_w = np.diff(m["furrows_m"])
        _MAPS[name] = dict(leaf=leaf, cover=np.where(np.isfinite(cover), cover, 0.0), good=good,
                           leaf_median=float(np.median(leaf[planted])), cover_median=med_cover,
                           bed_spacing_m=float(geo["bed_spacing_m"]), step_m=float(m["step_m"]), bed_width_m=bed_w,
                           date=str(m["date"]),
                           acres=float((good * bed_w[:, None]).sum() * float(m["step_m"]) / 4046.86))
        # area (m2) of field inside every window of beds x spots, from one cumulative sum (for cutting pieces)
        cell = good * bed_w[:, None] * float(m["step_m"])
        _MAPS[name]["area_cumsum"] = np.pad(cell.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    return _MAPS[name]


def window_acres(f, k0, k1, j0, j1):
    """Acres of field in beds k0..k1-1 x spots j0..j1-1."""
    A = f["area_cumsum"]
    return float((A[k1, j1] - A[k0, j1] - A[k1, j0] + A[k0, j0]) / 4046.86)


def split_regions(key):
    """The spatial 80/20 split of real field `key` ('f1' or 'f2'), the same for all its flights (one bed grid).
    The TEST region is one end of the field holding SPLIT_TEST_SHARE of its crop, cut across the field's longer
    side so it is compact (a band of beds on a wide field, the same stretch of every bed on a long one). The
    TRAINING region is the rest, minus a SPLIT_BUFFER_M strip next to the test region (spatial validation with a
    buffer: neighbouring plants look alike, Roberts et al. 2017). Of the field's two ends, the one whose plant
    NDVI and canopy cover (all flights) are closest to the whole field's is held out, so the test region is
    typical of the field. Returns {'train': (k0, k1, j0, j1), 'test': (...), 'axis', 'end', shares ...}."""
    if key in _SPLITS:
        return _SPLITS[key]
    maps = [m for m in available_maps() if m.startswith(key + "_")]
    if not maps:
        raise SystemExit(f"no maps for {key}: run `python -m amiga_scout extract-all` first")
    fs = [load_field(m) for m in maps]
    good = np.any([f["good"] for f in fs], axis=0)
    f0 = fs[0]
    nK, nJ = good.shape
    cell = good * f0["bed_width_m"][:, None] * f0["step_m"]
    ks, js = np.flatnonzero(good.any(1)), np.flatnonzero(good.any(0))
    across = float(f0["bed_width_m"][ks[0]:ks[-1] + 1].sum())
    along = (js[-1] - js[0] + 1) * f0["step_m"]
    axis = 0 if across >= along else 1                       # cut across the longer side
    prof = cell.sum(1 - axis)
    spacing = f0["bed_spacing_m"] if axis == 0 else f0["step_m"]
    nbuf = int(math.ceil(SPLIT_BUFFER_M / spacing))
    cum = np.concatenate([[0.0], np.cumsum(prof)])          # crop area before index i
    total = cum[-1]
    n = nK if axis == 0 else nJ
    c_high = int(np.searchsorted(cum, (1.0 - SPLIT_TEST_SHARE) * total))   # test = [c_high, n)
    c_low = int(np.searchsorted(cum, SPLIT_TEST_SHARE * total))            # test = [0, c_low)
    cand = {"high": ((c_high, n), (0, c_high - nbuf)), "low": ((0, c_low), (c_low + nbuf, n))}

    def rect(lo, hi):
        return (lo, hi, 0, nJ) if axis == 0 else (0, nK, lo, hi)

    def distance(lo, hi):
        """How far a region's plant NDVI and cover medians are from the whole field's (in IQRs, mean of flights)."""
        k0, k1, j0, j1 = rect(lo, hi)
        d = 0.0
        for f in fs:
            for arr in (f["leaf"], f["cover"]):
                ok = f["good"] & np.isfinite(arr)
                sub = arr[k0:k1, j0:j1][ok[k0:k1, j0:j1]]
                q1, med, q3 = np.percentile(arr[ok], [25, 50, 75])
                d += abs(float(np.median(sub)) - float(med)) / max(float(q3 - q1), 1e-6)
        return d / len(fs)

    end = min(cand, key=lambda e: distance(*cand[e][0]))
    (t0, t1), (r0, r1) = cand[end]
    out = dict(test=rect(t0, t1), train=rect(r0, r1), axis=axis, end=end, buffer_cells=nbuf,
               test_share=float((cum[t1] - cum[t0]) / total), train_share=float((cum[r1] - cum[r0]) / total),
               test_distance=distance(t0, t1), other_end_distance=distance(*cand["low" if end == "high" else "high"][0]))
    _SPLITS[key] = out
    return out


def field_map(name):
    """The map a mission on field `name` is laid on: the whole map (load_field), or a tiled region (tiled_field)."""
    return tiled_field(name) if tile_of(name) != (1, 1) else load_field(name)


_TILED = {}


def tiled_field(name):
    """A big field made of real drone data only: the region of `name` (or the whole map) laid out in a grid of
    copies, tile_of(name) = (across the beds, along them); every other copy is mirrored, so beds and texture run on
    across the seams. Copies side by side across the beds never break a bed; copies along the beds should meet at a
    straight cut (a region's edge next to the buffer), or a bed would get a bare gap where the field's ragged ends
    meet. The medians stay the whole field's (like a piece). Same keys as load_field."""
    base, region, _ = parse_name(name)
    na, nb = tile_of(name)
    key = (base, region, na, nb)
    if key in _TILED:
        return _TILED[key]
    f = load_field(base)
    k0, k1, j0, j1 = split_regions(field_key(base))[region] if region else (0, f["leaf"].shape[0], 0, f["leaf"].shape[1])
    bw = f["bed_width_m"][k0:k1]

    def tile(arr):
        sub = arr[k0:k1, j0:j1]
        rows = []
        for i in range(na):
            blocks = []
            for j in range(nb):
                b = sub[::-1] if i % 2 else sub
                blocks.append(b[:, ::-1] if j % 2 else b)
            rows.append(np.concatenate(blocks, axis=1))
        return np.concatenate(rows, axis=0)
    good = tile(f["good"])
    bed_w = np.concatenate([bw[::-1] if i % 2 else bw for i in range(na)])
    cell = good * bed_w[:, None] * f["step_m"]
    out = dict(f, leaf=tile(f["leaf"]), cover=tile(f["cover"]), good=good, bed_width_m=bed_w,
               acres=float(cell.sum() / 4046.86), area_cumsum=np.pad(cell.cumsum(0).cumsum(1), ((1, 0), (1, 0))),
               tiles=(na, nb))
    _TILED[key] = out
    return out


def region_bounds(name):
    """(k0, k1, j0, j1) of the region a field name refers to, or None (the whole map - or a tiled field, whose
    map is already made of the region)."""
    region = region_of(name)
    return split_regions(field_key(name))[region] if region and tile_of(name) == (1, 1) else None


def draw_piece(f, acres, rng, bounds=None, min_share=0.9, tries=500):
    """A random piece of a real field holding about `acres` of crop: whole beds x a stretch along them, a random
    shape from 1:2 to 2:1, at a random place (inside `bounds` = (k0, k1, j0, j1), a region of the field, if given).
    The window starts at the asked size and grows over the tries where the field is patchy (an irregular outline),
    until the crop inside it is within min_share .. 2 - min_share of the asked size; the real outline stays inside
    the piece. Returns (first bed, last bed + 1, first spot, last spot + 1, acres of field in it)."""
    target = acres * 4046.86
    bk0, bk1, bj0, bj1 = bounds or (0, f["leaf"].shape[0], 0, f["leaf"].shape[1])
    nK, nJ = bk1 - bk0, bj1 - bj0
    for t in range(tries):
        grow = 1.0 + 2.0 * t / tries
        ratio = math.exp(rng.uniform(math.log(0.5), math.log(2.0)))      # length along the beds / width across
        width = math.sqrt(target * grow / ratio)
        nk = min(nK, max(1, round(width / f["bed_spacing_m"])))
        nj = min(nJ, max(6, round(ratio * width / f["step_m"])))
        k0, j0 = bk0 + int(rng.integers(0, nK - nk + 1)), bj0 + int(rng.integers(0, nJ - nj + 1))
        acres_in = window_acres(f, k0, k0 + nk, j0, j0 + nj)
        if min_share * target <= acres_in * 4046.86 <= (2.0 - min_share) * target:
            return k0, k0 + nk, j0, j0 + nj, acres_in
    raise SystemExit(f"could not cut a {acres:g}-acre piece from this field or region (is it smaller than that?)")
