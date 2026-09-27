"""The survey on disk (fields, flights, file paths) and pixel-level helpers."""
from __future__ import annotations

import numpy as np

from ..paths import DRONE_ROOT


ROOT = DRONE_ROOT
FIELDS = {
    "f1": dict(name="Field 1 (Onions)", polygon=f"{ROOT}/Shapefiles/Field1/Polygon", ortho="{d}_Onions.tif",
               samples=f"{ROOT}/Dataset/Field data/dataset_field_onions.csv",
               dem_chunks={"02": 3, "03": 4, "04": 5, "05": 6, "06": 7}),
    "f2": dict(name="Field 2 (Ashley)", polygon=f"{ROOT}/Shapefiles/Field2/polygonUTM", ortho="{d}_Onions_Ashley.tif",
               samples=f"{ROOT}/Dataset/Field data/dataset_field_onions_ashley.csv",
               dem_chunks={"01": 9, "02": 10, "03": 11, "04": 12, "05": 13, "06": 14}),
}
DATES = {"01": "2024-01-31", "02": "2024-02-14", "03": "2024-03-07", "04": "2024-03-18", "05": "2024-04-04",
         "06": "2024-04-17"}
STEP_M = 3.048 / 6          # one plant spot: a 10-ft block holds 6
NODATA = -32767.0
CRS = "EPSG:32617"
# --------------------------------------------------------------------------------------------- bed geometry
PROFILE_BIN_M = 0.02        # across-bed profile resolution


# --------------------------------------------------------------------------------------------- helpers
def _polygon(path):
    import shapefile
    shp = shapefile.Reader(path)
    return [s.__geo_interface__ for s in shp.shapes()], np.array(shp.shapes()[0].points)


def otsu(values, lo=-0.2, hi=1.0, bins=256):
    h, e = np.histogram(values, bins=bins, range=(lo, hi))
    c = (e[:-1] + e[1:]) / 2
    w0 = np.cumsum(h); w1 = w0[-1] - w0
    m0 = np.cumsum(h * c) / np.maximum(w0, 1); m1 = (np.sum(h * c) - np.cumsum(h * c)) / np.maximum(w1, 1)
    return float(c[np.argmax(w0 * w1 * (m0 - m1) ** 2)])


def _ndvi(red, nir):
    ok = (red != NODATA) & (nir != NODATA) & (red + nir > 0)
    return np.where(ok, (nir - red) / np.where(ok, nir + red, 1), np.nan)


def ortho_path(field, date):
    return f"{ROOT}/Orthomosaic/" + FIELDS[field]["ortho"].format(d=date)


def _sample_pixels(field, date, n_strips=48, rows=32, every=3):
    """A field-wide sample of pixels (every `every`-th pixel of n_strips thin strips across the whole field, inside
    the outline): easting/northing and NDVI."""
    import rasterio
    from rasterio.windows import Window
    from rasterio.features import geometry_mask
    geo, pts = _polygon(FIELDS[field]["polygon"])
    E, N, P = [], [], []
    with rasterio.open(ortho_path(field, date)) as r:
        px = r.res[0]
        c0 = max(0, int((pts[:, 0].min() - r.transform.c) / px)); c1 = min(r.width, int((pts[:, 0].max() - r.transform.c) / px))
        r0 = max(0, int((r.transform.f - pts[:, 1].max()) / px)); r1 = min(r.height, int((r.transform.f - pts[:, 1].min()) / px))
        for y in np.linspace(r0, r1 - rows, n_strips).astype(int):
            w = Window(c0, y, c1 - c0, rows)
            red, nir = r.read([2, 4], window=w).astype("float32")
            nd = _ndvi(red, nir)
            inside = ~geometry_mask(geo, out_shape=nd.shape, transform=r.window_transform(w))
            rr, cc = np.nonzero(np.isfinite(nd) & inside)
            rr, cc = rr[::every], cc[::every]
            E.append(r.transform.c + px * (c0 + cc + 0.5)); N.append(r.transform.f - px * (y + rr + 0.5)); P.append(nd[rr, cc])
    return np.concatenate(E), np.concatenate(N), np.concatenate(P).astype("float32")


def _profile(u_or_none, v, plant, lo, n_bins):
    """Plant share across the beds (bins of PROFILE_BIN_M in v) and the pixel count per bin."""
    idx = np.clip(((v - lo) / PROFILE_BIN_M).astype(int), 0, n_bins - 1)
    n = np.bincount(idx, minlength=n_bins)
    return np.bincount(idx, weights=plant, minlength=n_bins) / np.maximum(n, 1), n
