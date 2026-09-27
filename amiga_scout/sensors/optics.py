"""The physics of the cameras: plain formulas, each with where it comes from.

    stereo depth noise   sigma_z = z^2 * sigma_d / (f_px * B)          Keselman et al. 2017, eq. 2; Intel tuning
                                                                       white paper (sigma_d ~0.05-0.1 px, textured)
    minimum distance     datasheet table (it scales with the depth resolution; ~ f_px * B / 126 disparities)
    ground resolution    pixel footprint = z * 2 tan(FOV/2) / width   (or z * pixel pitch / lens focal length)
    defocus blur         blur circle = (f/N) * f * |z - z_focus| / (z * (z_focus - f))     thin lens
    motion blur          speed * exposure time / ground resolution
"""
from __future__ import annotations

import math


def parse_resolution(res):
    w, h = (int(v) for v in str(res).lower().split("x"))
    return w, h


def depth_fov_deg(cam, res):
    """(horizontal, vertical) depth field of view of this depth resolution: 16:9 modes use the HD field of view,
    4:3 modes the VGA one (datasheet table 3-42)."""
    w, h = parse_resolution(res)
    return cam.fov_hd_deg if abs(w / h - 16 / 9) < 0.1 else cam.fov_vga_deg


def focal_px(width_px, hfov_deg):
    """Focal length in pixels of a camera `width_px` wide with horizontal field of view `hfov_deg`."""
    return 0.5 * width_px / math.tan(math.radians(hfov_deg) / 2)


def depth_focal_px(cam, res):
    return focal_px(parse_resolution(res)[0], depth_fov_deg(cam, res)[0])


def min_z(cam, res):
    """Closest distance (m) that still gives depth at this resolution (datasheet); closer = no depth at all."""
    if res in cam.min_z_m:
        return cam.min_z_m[res]
    return depth_focal_px(cam, res) * cam.baseline_m / 126.0          # the D4 processor's disparity search


def depth_rms(cam, res, z, subpixel):
    """Depth noise (m, RMS) of one depth pixel at distance z (m): grows with the distance squared."""
    return z * z * subpixel / (depth_focal_px(cam, res) * cam.baseline_m)


def depth_gsd(cam, res, z):
    """Size (m) of one depth pixel on the plant at distance z."""
    w, _ = parse_resolution(res)
    return z * 2 * math.tan(math.radians(depth_fov_deg(cam, res)[0]) / 2) / w


def lens_gsd(pixel_m, focal_m, z):
    """Size (m) of one pixel on the plant at distance z, for a sensor pixel `pixel_m` behind a `focal_m` lens."""
    return z * pixel_m / focal_m


def lens_fov_deg(sensor_mm, focal_m):
    return tuple(2 * math.degrees(math.atan(s / 2e3 / focal_m)) for s in sensor_mm)


def defocus_px(focal_m, fnumber, focus_m, z, pixel_m):
    """Blur circle (pixels) of a point at distance z when the lens is focused at focus_m (thin lens)."""
    aperture = focal_m / fnumber
    blur = aperture * focal_m * abs(z - focus_m) / (z * (focus_m - focal_m))
    return blur / pixel_m


def depth_of_field(focal_m, fnumber, focus_m, coc_m):
    """(near, far) limits (m) inside which blur stays under the circle of confusion coc_m."""
    hyper = focal_m ** 2 / (fnumber * coc_m) + focal_m
    near = focus_m * (hyper - focal_m) / (hyper + focus_m - 2 * focal_m)
    far = focus_m * (hyper - focal_m) / (hyper - focus_m) if focus_m < hyper else math.inf
    return near, far


def pixel_ndvi_noise(ndvi, snr_db_nir, snr_db_red, nir_level=0.8):
    """NDVI noise of ONE pixel from sensor noise (shot-noise limited): the exposure puts the NIR signal at
    nir_level of full scale; red comes in weaker (a leaf reflects much less red), so it is the noisier band."""
    red_over_nir = (1 - ndvi) / (1 + ndvi)
    s_n = 10 ** (-snr_db_nir / 20) / math.sqrt(nir_level)
    s_r = 10 ** (-snr_db_red / 20) / math.sqrt(max(nir_level * red_over_nir, 1e-3))
    k = 2 * red_over_nir / (1 + red_over_nir) ** 2                 # d NDVI / d log(band) for both bands
    return k * math.hypot(s_n, s_r)
