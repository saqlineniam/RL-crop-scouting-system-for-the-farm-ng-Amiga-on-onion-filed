"""One shot of one plant: what the cameras see from where the robot stopped, the quality numbers the robot can
compute on the spot, and the errors of the height and NDVI readings.

The cameras look straight down from CAMERA_HEIGHT_M, mounted with their wide side ACROSS the bed (to cover the
rows). For a plant h tall, the depth camera sees the canopy top at z = camera height - h, the NIR camera the middle
of the canopy. The plant sits dx along the row from the cameras (the robot changes this by moving back or forward)
and y across the bed (its row; the robot cannot change this), so the view angle is atan(sqrt(dx^2 + y^2) / z). A
plant near the edge of a camera's view is partly cut off; beyond it, that camera does not see it at all.

HEIGHT (D455):
    no depth at all        z below the minimum distance of the depth resolution (datasheet)
    depth quality          fill rate = field fill rate (Fan et al.) x the lab's quality-vs-distance log x the share
                           of the plant's top neighbours leave visible at this view angle x the share of leaves wide
                           enough for the stereo matcher (seedlings far away give no depth)
    thin leaf tips         a tip narrower than the stereo matcher resolves (a few pixels: 7x7 census, Keselman et al.)
                           gives no depth, so the plant looks shorter; the lab calibrates the typical loss out, the
                           plant-to-plant part stays
    hidden top             off-centre views may hide the tallest tip behind neighbours (a downward error)
    this viewpoint         the D455's systematic depth error (Servi et al., larger closer than 0.5 m) + which leaf
                           tip reads as the top: fixed for a viewpoint - shooting again from the SAME spot does not
                           average it out, a shot from another spot does
    random                 stereo noise (z^2 law) over the plant's top pixels and frames + leaf sway, averaged over
                           the time a careful shot takes
NDVI (JAI AD-130GE, red + NIR through one lens):
    NIR quality            share of the plant's leaf pixels that are clean: not mixed with soil at leaf edges (blur
                           from pixel size, focus and leaf motion vs the leaf's width) and not glared
    mixed edge pixels      pull NDVI toward the soil's (onion leaves are thin tubes, 4-20 mm); the leaf mask is
                           eroded, so this matters only when leaves are hardly wider than the blur
    glare                  specular glare of the scan light on waxy leaves near one spot: spectrally flat, so NDVI
                           reads low there (Krafft et al. 2024); moving off it helps
    this viewpoint         leaf angle / illumination geometry (Zhang et al. 2022, Huang et al. 2018): fixed for a
                           viewpoint, like the height one
    calibration            one offset for the whole mission (reference-panel calibration, Stamford et al. 2023)
    random                 sensor noise per pixel (S/N from the datasheet) over the clean leaf pixels - tiny
"""
from __future__ import annotations

import math

from .optics import defocus_px, depth_fov_deg, depth_gsd, depth_rms, lens_gsd, min_z, pixel_ndvi_noise
from .profiles import D455_DEPTH, D455_RGB, JAI_AD130GE, lab_depth_quality

MIXED_PIXEL_SOIL_SHARE = 0.5     # a leaf-edge pixel is, on average, half leaf and half soil
TOP_SECTION_M = 0.05             # the canopy top the height is read from: the top 5 cm ...
TOP_LEAVES = 3                   # ... of about three leaves
VISIBLE_LEAF_M = 0.25            # visible length of each leaf in the NIR image
YOLO_PLANT_SPAN_M = 0.15         # plant width YOLO has to find (only if YOLO_CAN_MISS)
PLANT_RADIUS_M = 0.05            # a plant's top is fully in a camera's view once it is this far inside the edge


def view_angle(dx, y, z):
    return math.atan2(math.hypot(dx, y), max(z, 1e-3))


def _in_view(half_along, half_across, dx, y):
    """Share of the plant inside a camera's view (0 = outside it)."""
    along = min(1.0, max(0.0, (half_along - abs(dx)) / PLANT_RADIUS_M + 0.5))
    across = min(1.0, max(0.0, (half_across - abs(y)) / PLANT_RADIUS_M + 0.5))
    return along * across


def depth_footprint(cfg, z):
    """(along the row, across the bed) width (m) the D455 depth camera sees at distance z (wide side across)."""
    h, v = depth_fov_deg(D455_DEPTH, cfg.depth_resolution)
    return 2 * z * math.tan(math.radians(v) / 2), 2 * z * math.tan(math.radians(h) / 2)


def nir_footprint(cfg, z):
    """(along the row, across the bed) width (m) the NIR camera sees at distance z (wide side across)."""
    f = cfg.nir_lens_focal_mm / 1e3
    across_mm, along_mm = JAI_AD130GE.sensor_mm
    return z * along_mm / 1e3 / f, z * across_mm / 1e3 / f


def rgb_footprint(z):
    """(along the row, across the bed) width (m) the D455 RGB camera sees at distance z (wide side across)."""
    h, v = D455_RGB.fov_deg
    return 2 * z * math.tan(math.radians(v) / 2), 2 * z * math.tan(math.radians(h) / 2)


def visible_top_share(cfg, theta):
    """Share of the plant's top the depth camera sees at view angle theta (neighbouring plants hide the rest)."""
    return 0.5 ** ((theta / math.radians(cfg.occlusion_half_angle_deg)) ** 2)


def resolvable_share(cfg, z, leaf_w_m):
    """Share of the plant's leaves wide enough for the stereo matcher at distance z: leaves narrower than about
    half its smallest feature are invisible to depth, leaves 1.5x wider are fully seen (seedlings seen from far
    away may give no depth at all)."""
    w_min = cfg.stereo_min_feature_px * depth_gsd(D455_DEPTH, cfg.depth_resolution, z)
    return min(1.0, max(0.0, leaf_w_m / w_min - 0.5))


def depth_quality(cfg, z, dx, y, leaf_w_m):
    """Fill rate on the plant (0-1): what the robot reads live as depth quality. 0 when closer than the minimum
    distance of the depth resolution, when the plant is outside the depth camera's view, or when the leaves are too
    thin for the stereo matcher at this distance."""
    if z < min_z(D455_DEPTH, cfg.depth_resolution):
        return 0.0
    along, across = depth_footprint(cfg, z)
    return (cfg.depth_fill_field * min(1.0, lab_depth_quality(z)) * visible_top_share(cfg, view_angle(dx, y, z))
            * resolvable_share(cfg, z, leaf_w_m) * _in_view(along / 2, across / 2, dx, y))


def nir_quality(cfg, z_mid, dx, y, leaf_w_m, glare_amp, glare_pos_m):
    """(quality, mixed share, glare share) of the red/NIR image: quality = share of clean leaf pixels of the plant
    (0 when it is outside the NIR camera's view)."""
    f = cfg.nir_lens_focal_mm / 1e3
    px = JAI_AD130GE.pixel_m
    gsd = lens_gsd(px, f, z_mid)
    blur_defocus = defocus_px(f, cfg.nir_lens_fnumber, cfg.nir_focus_m, z_mid, px)
    blur_motion = cfg.leaf_sway_speed_mps * cfg.wind * cfg.nir_exposure_ms / 1e3 / gsd
    blur_m = gsd * math.sqrt(1.0 + blur_defocus ** 2 + blur_motion ** 2)
    mixed = min(1.0, blur_m / leaf_w_m)
    glare = cfg.glare_max_share * glare_amp * math.exp(-0.5 * ((dx - glare_pos_m) / cfg.glare_width_m) ** 2)
    along, across = nir_footprint(cfg, z_mid)
    return (1.0 - mixed) * (1.0 - glare) * _in_view(along / 2, across / 2, dx, y), mixed, glare


def viewpoint_error(g1, g2, dx, corr_m):
    """A smooth random function of the camera position (unit variance): the same spot gives the same error,
    a spot corr_m away gives an unrelated one."""
    a = dx * (math.pi / 2) / corr_m
    return g1 * math.cos(a) + g2 * math.sin(a)


def averaging(cfg, careful):
    """(frames, independent leaf-sway samples) of one shot: a quick shot is one frame; a careful shot averages
    the frames of careful_average_s (leaf sway is correlated over sway_corr_s)."""
    if not careful:
        return 1.0, 1.0
    return 1.0 + cfg.careful_average_s * cfg.frame_rate_hz, 1.0 + cfg.careful_average_s / cfg.sway_corr_s


def height_shot(cfg, rng, h_cm, z, theta, fill, leaf_w_m, stage_w_m, view, careful):
    """One height reading (cm) and the error the robot attributes to it; (None, None) when there is no depth."""
    if fill <= 0.0:
        return None, None
    res = cfg.depth_resolution
    gsd = depth_gsd(D455_DEPTH, res, z)
    w_min = cfg.stereo_min_feature_px * gsd                         # thinnest leaf the matcher resolves
    loss = cfg.leaf_tip_taper_cm * min(1.0, w_min / leaf_w_m)
    loss_expected = cfg.leaf_tip_taper_cm * min(1.0, w_min / stage_w_m)
    hidden_sd = cfg.canopy_top_sd_cm * (1.0 - visible_top_share(cfg, theta))
    hidden = hidden_sd * abs(float(rng.standard_normal()))
    view_sd = math.hypot(cfg.depth_systematic_mm / 10 * max(1.0, (0.5 / z) ** 2), cfg.canopy_top_sd_cm)
    frames, sway_samples = averaging(cfg, careful)
    pix = 100 * depth_rms(D455_DEPTH, res, z / math.cos(theta), cfg.depth_subpixel_rms)
    n_px = max(1.0, fill * TOP_LEAVES * leaf_w_m * TOP_SECTION_M / gsd ** 2)
    rand = math.hypot(pix / math.sqrt(n_px * frames), cfg.leaf_sway_cm * cfg.wind / math.sqrt(sway_samples))
    reading = h_cm - (loss - loss_expected) - hidden + view_sd * view + rand * float(rng.standard_normal())
    believed = math.sqrt(rand ** 2 + view_sd ** 2 + (cfg.leaf_width_cv * loss_expected) ** 2 + hidden_sd ** 2)
    return reading, believed


def ndvi_shot(cfg, rng, ndvi, quality, mixed, glare, z_mid, leaf_w_m, view, calibration, careful):
    """One NDVI reading and the error the robot attributes to it."""
    # the leaf mask is eroded to drop edge pixels (usual practice); what is left mixed grows ~ mixed^2: little
    # when leaves are many pixels wide, a lot when they are hardly wider than the blur
    mix_bias = MIXED_PIXEL_SOIL_SHARE * mixed ** 2 * (ndvi - cfg.soil_ndvi)
    glare_bias = cfg.glare_ndvi_drop * glare
    frames, _ = averaging(cfg, careful)
    gsd = lens_gsd(JAI_AD130GE.pixel_m, cfg.nir_lens_focal_mm / 1e3, z_mid)
    n_clean = max(1.0, quality * TOP_LEAVES * leaf_w_m * VISIBLE_LEAF_M / gsd ** 2)
    if quality <= 0.0:                               # the plant is not in the NIR image: no NDVI from this shot
        return None, None
    pix = pixel_ndvi_noise(min(max(ndvi, 0.05), 0.95), JAI_AD130GE.snr_db["nir"], JAI_AD130GE.snr_db["colour"])
    rand = pix / math.sqrt(n_clean * frames)
    reading = ndvi - mix_bias - glare_bias + cfg.ndvi_view_sd * view + calibration + rand * float(rng.standard_normal())
    expected_mix = MIXED_PIXEL_SOIL_SHARE * mixed ** 2 * (cfg.prior_ndvi - cfg.soil_ndvi)
    believed = math.sqrt(rand ** 2 + cfg.ndvi_view_sd ** 2 + expected_mix ** 2 + glare_bias ** 2)
    return reading, believed


def p_detect(cfg, z):
    """Chance YOLO finds the plant (only used if YOLO_CAN_MISS): the plant must span enough RGB pixels."""
    if not cfg.yolo_can_miss:
        return 1.0
    gsd = z * 2 * math.tan(math.radians(D455_RGB.fov_deg[0]) / 2) / D455_RGB.width_px
    span = YOLO_PLANT_SPAN_M / gsd
    return 1.0 / (1.0 + math.exp(-(span - cfg.yolo_min_plant_px) / (0.25 * cfg.yolo_min_plant_px)))


def shot_statistics(cfg, rng, height_cm, leaf_w_mm, n=2000, offsets=None):
    """One plant measured once, at one growth stage, as the model gives it, positioned like the hand-written rules:
    'quick' = one frame from where the robot stopped; 'careful' = shot from the position with the best depth
    quality seen on the way in, frames averaged, and - if that NIR image had too few clean leaf pixels - a second
    shot from the best NIR position seen, fused with the first. Returns, per mode: height error (bias, SD, cm),
    share of plants without any height, NDVI error (bias, SD), mean depth and NIR quality of the first shot."""
    offsets = offsets or [-0.3, -0.2, -0.1, 0.0]          # positions the robot passed over while driving up
    stage_w = leaf_w_mm / 1e3
    out = {}
    for mode in ("quick", "careful"):
        careful = mode == "careful"
        dh, dn, no_depth, qds, qns = [], [], 0, [], []
        for _ in range(n):
            h = height_cm * min(1.5, max(0.5, 1 + 0.12 * float(rng.standard_normal())))
            w = stage_w * min(1.8, max(0.4, 1 + cfg.leaf_width_cv * float(rng.standard_normal())))
            centre = float(rng.normal(0, cfg.plant_spot_sd_m))
            across = float(rng.uniform(0, cfg.row_spacing_across_m / 2))
            glare_pos = cfg.lamp_offset_m / 2 + float(rng.normal(0, cfg.glare_width_m))
            g = rng.standard_normal(4)
            x0 = cfg.arrival_offset_m + float(rng.normal(0, cfg.arrival_jitter_m))
            z_top, z_mid = cfg.camera_height_m - h / 100, cfg.camera_height_m - h / 200
            amp = float(rng.uniform())

            def quality(x):
                qd = depth_quality(cfg, z_top, x - centre, across, w)
                qn, mixed, glare = nir_quality(cfg, z_mid, x - centre, across, w, amp, glare_pos)
                return qd, qn, mixed, glare

            def shoot(x):
                dx = x - centre
                qd, qn, mixed, glare = quality(x)
                yh, sh = height_shot(cfg, rng, h, z_top, view_angle(dx, across, z_top), qd, w, stage_w,
                                     viewpoint_error(g[0], g[1], dx, cfg.view_corr_m), careful)
                yn, sn = ndvi_shot(cfg, rng, 0.75, qn, mixed, glare, z_mid, w,
                                   viewpoint_error(g[2], g[3], dx, cfg.view_corr_m), 0.0, careful)
                return qd, qn, yh, sh, yn, sn

            if careful:
                seen = {o: quality(x0 + o) for o in offsets}
                noisy = {o: q[0] + float(rng.normal(0, cfg.preview_noise_moving)) for o, q in seen.items()}
                x = x0 + max(noisy, key=noisy.get)
            else:
                x = x0
            qd, qn, yh, sh, yn, sn = shoot(x)
            if careful and qn < cfg.nir_reshoot_below:
                noisy = {o: seen[o][1] + float(rng.normal(0, cfg.preview_noise_moving)) for o in offsets if x0 + o != x}
                _, _, yh2, sh2, yn2, sn2 = shoot(x0 + max(noisy, key=noisy.get))
                yh, sh = _fuse(yh, sh, yh2, sh2)
                yn, sn = _fuse(yn, sn, yn2, sn2)
            if yh is None:
                no_depth += 1
            else:
                dh.append(yh - h)
            if yn is not None:
                dn.append(yn - 0.75)
            qds.append(qd)
            qns.append(qn)
        dh, dn = dh or [float("nan")], dn or [float("nan")]
        out[mode] = dict(height_bias=float(sum(dh) / len(dh)), height_sd=float(_sd(dh)), no_depth=no_depth / n,
                         ndvi_bias=float(sum(dn) / len(dn)), ndvi_sd=float(_sd(dn)),
                         depth_quality=float(sum(qds) / n), nir_quality=float(sum(qns) / n))
    return out


def _fuse(a, sa, b, sb):
    """Precision-weighted mean of two readings (either may be missing)."""
    if a is None:
        return b, sb
    if b is None:
        return a, sa
    w_a, w_b = 1 / sa ** 2, 1 / sb ** 2
    return (a * w_a + b * w_b) / (w_a + w_b), (w_a + w_b) ** -0.5


def _sd(v):
    m = sum(v) / len(v)
    return math.sqrt(sum((a - m) ** 2 for a in v) / max(len(v) - 1, 1))
