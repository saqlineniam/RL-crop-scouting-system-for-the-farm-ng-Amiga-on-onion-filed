"""All settings in one place: the values to measure in the lab (top block), the per-mission settings
(`Cfg`, each overridable with --set name=value), the field types and the stress tests."""
from __future__ import annotations

from dataclasses import dataclass


# =====================================================================================
#  EDIT THESE IN THE LAB
# =====================================================================================
# ---- Amiga power ---------------------------------------------------------------------
PACK_VOLTAGE_V = 44.0          # SPEC: 12S Li-ion pack, 44 V nominal
PACK_CAPACITY_AH = 15.0        # SPEC: 15 Ah per pack  (=> ~660 Wh per pack)
PACKS_ONBOARD = 2              # SPEC / LAB: the Amiga carries two packs at once
SPARE_PACK_SETS = 1            # PLACEHOLDER: charged pairs waiting at base (1 = one swap allowed)
USABLE_CAPACITY_FRAC = 0.85    # PLACEHOLDER: aging, cold, BMS cut-off (1.0 = brand-new pack)
RETURN_TRIGGER_PCT = 20.0      # YOUR RULE: at/below this charge, drive home now
ARRIVE_MIN_PCT = 5.0           # PLACEHOLDER: never plan a trip that arrives home below this
AMIGA_RUNTIME_H = 8.0          # LAB: one charge runs the platform ~8 h
LOAD_FACTOR = 1.1              # PLACEHOLDER: extra motor draw from our payload (1.0 = none)
HOTEL_POWER_W = 45.0           # PLACEHOLDER: computer + D455 + NIR camera + radio, always on
LIGHT_POWER_W = 40.0           # PLACEHOLDER: scan light - LAB: always on, except during bug-zapper stops
                               #   (the zapper has its own light, counted in ZAPPER_POWER_W)
# motor draw while driving, derived so that driving on one charge lasts AMIGA_RUNTIME_H
# (re-derived only when you edit the values here; with --set, give drive_power_w directly):
DRIVE_POWER_W = (PACK_VOLTAGE_V * PACK_CAPACITY_AH * PACKS_ONBOARD * USABLE_CAPACITY_FRAC / AMIGA_RUNTIME_H
                 - HOTEL_POWER_W) * LOAD_FACTOR
PROFESSOR_ACRES_PER_CHARGE = 30.0  # LAB (rough, NO load): both packs run ~30 acres - a reference only

# ---- bug zapper ----------------------------------------------------------------------
ZAPPER_ENABLED = True          # LAB: the platform carries a bug zapper (set False to ignore it)
ZAP_EVERY_MIN = 30.0           # LAB: after every 30 min of field work the platform stops ...
ZAP_DURATION_MIN = 15.0        # LAB: ... and zaps for 15 min where it stands
ZAPPER_POWER_W = 60.0          # PLACEHOLDER: zapper draw while running

# ---- Amiga motion --------------------------------------------------------------------
MAX_SPEED_MPS = 3.6 * 0.44704  # SPEC: top speed 3.6 mph (1.61 m/s) - a ceiling, not the speed it always drives at
ROW_SPEED_MPS = 0.89           # PLACEHOLDER: speed between stops INSIDE a bed while scouting (~2 mph; with a stop
                               #   every 10 ft it never gets near top speed anyway - see HOP_OVERHEAD_S)
TRAVEL_SPEED_MPS = 1.34        # PLACEHOLDER: speed when driving WITHOUT stopping (~3 mph): on to the end of a pass,
                               #   along the headland, to and from base for a battery swap
                               # (both are capped at MAX_SPEED_MPS; set your real operating speeds here)
HOP_OVERHEAD_S = 2.0           # PLACEHOLDER: speeding up and slowing down for each stop
STOP_SETTLE_S = 1.0            # PLACEHOLDER: time to stop and let the camera settle
NUDGE_STEP_M = 0.10            # PLACEHOLDER: one slight FORWARD/BACKWARD adjustment at a plant
NUDGE_SPEED_MPS = 0.15         # PLACEHOLDER
ROW_TURN_S = 25.0              # PLACEHOLDER: headland turn into the next pass
HEADLAND_M = 5.0               # PLACEHOLDER: distance driven outside the field when leaving a pass
SWAP_TIME_S = 900.0            # PLACEHOLDER: arrive, swap both packs, checks, leave (15 min)
                               # (no working-day limit: only the batteries limit a mission)

# ---- the cameras and what they see (amiga_scout/sensors; sources in docs/references.md) ----------------
#   height: Intel RealSense D455 depth, looking straight down     NDVI: JAI AD-130GE red + NIR (one lens)
#   `python -m amiga_scout cameras` shows the curves these give (distance limits, blur, errors)
CAMERA_HEIGHT_M = 1.5          # from `mount`: cameras above the ground, looking down. The lowest height at which every
                               #   camera sees the whole 1.4-m canopy across the bed at the tallest plants (65-70 cm)
                               #   and no plant is closer than the D455's minimum distance. (The platform was 1.1 m:
                               #   `--set camera_height_m=1.1 --set depth_resolution=848x480 --set nir_lens_focal_mm=6
                               #   --set nir_lens_fnumber=4 --set nir_focus_m=0.7` gives that old setup.)
DEPTH_RESOLUTION = "1280x720"  # from `mount`: sets Min-Z (SPEC table: 1280x720 0.52 m, 848x480 0.35 m, 640x360 0.26 m)
                               #   and pixel size; at 1.5 m the tallest plants are still ~0.8 m away, above Min-Z
DEPTH_SUBPIXEL_RMS = 0.08      # PAPER: stereo disparity noise, pixels (Intel tuning paper: 0.05-0.1 textured;
                               #   Keselman et al. 2017: ~0.1)
DEPTH_SYSTEMATIC_MM = 4.25     # PAPER: D455 systematic depth error, SD 500-1500 mm (Servi et al. 2021; larger closer)
DEPTH_FILL_FIELD = 0.90        # PAPER: depth fill rate on crops in the field (~90%, D435i: Fan et al.)
STEREO_MIN_FEATURE_PX = 3.0    # PLACEHOLDER: a leaf thinner than this many depth pixels gives no depth (the matcher
                               #   works on 7x7-pixel windows: Keselman et al. 2017) - so thin tips are missed
OCCLUSION_HALF_ANGLE_DEG = 25.0  # PLACEHOLDER: viewed this far off-centre, neighbours hide half of the plant's top
NIR_LENS_FOCAL_MM = 2.8        # from `mount`: C-mount lens on the NIR camera, the longest standard lens that covers the
                               #   canopy across the bed at 1.5 m (2.8 mm: ~82 x 66 deg field of view)
NIR_LENS_FNUMBER = 2.0         # from `mount`: widest aperture the prism allows (f/2: SPEC); sharp all season at 1.5 m
NIR_FOCUS_M = 1.31             # from `mount`: focused at the season's median distance to mid-canopy
NIR_EXPOSURE_MS = 2.0          # PLACEHOLDER: exposure with the scan light (SPEC range 0.011-31.8 ms)
FRAME_RATE_HZ = 30.0           # SPEC: JAI 31 fps, D455 up to 90 fps
CAREFUL_AVERAGE_S = 1.0        # PLACEHOLDER: a careful shot averages the frames of this long (a quick shot: 1 frame)
LEAF_WIDTH_MM = {"01": 5.0, "02": 6.0, "03": 8.0, "04": 9.0, "05": 10.0, "06": 11.0}
                               # PLACEHOLDER: onion leaf width per flight date (PAPER: onion leaves are hollow tubes
                               #   4-20 mm wide); synthetic fields use LEAF_WIDTH_MM_DEFAULT
LEAF_WIDTH_MM_DEFAULT = 8.0
LEAF_WIDTH_CV = 0.2            # PLACEHOLDER: plant-to-plant variation of leaf width
LEAF_TIP_TAPER_CM = 8.0        # PLACEHOLDER: length of the leaf tip that narrows to a point
CANOPY_TOP_SD_CM = 0.8         # PLACEHOLDER: which leaf tip reads as "the top" from one viewpoint
LEAF_SWAY_CM = 0.5             # PLACEHOLDER: leaf-tip movement between frames on a calm day (x wind)
LEAF_SWAY_SPEED_MPS = 0.03     # PLACEHOLDER: leaf speed during an exposure (motion blur; x wind)
SWAY_CORR_S = 0.5              # PLACEHOLDER: leaf sway is correlated over this long (limits what averaging gains)
SOIL_NDVI = 0.15               # PLACEHOLDER: NDVI of the sandy soil as the robot sees it (edge pixels mix toward it)
NDVI_VIEW_SD = 0.02            # PLACEHOLDER: NDVI error from leaf angle / illumination geometry of one viewpoint
                               #   (PAPER: leaf NDVI changes strongly with leaf angle - Zhang et al. 2022)
VIEW_CORR_M = 0.15             # PLACEHOLDER: move the cameras this far and a viewpoint's errors are unrelated
GLARE_MAX_SHARE = 0.3          # PLACEHOLDER: share of leaf pixels glared by the scan light at the worst spot
GLARE_NDVI_DROP = 0.15         # PLACEHOLDER: NDVI drop of glared pixels (glare is spectrally flat: Krafft et al. 2024)
LAMP_OFFSET_M = 0.10           # PLACEHOLDER: scan light this far ahead of the NIR camera -> glare near half of that
GLARE_WIDTH_M = 0.06           # PLACEHOLDER: size of the glare spot along the row
NDVI_CALIBRATION_SD = 0.01     # PLACEHOLDER: NDVI offset of one mission's calibration (reference panel: Stamford 2023)

# ---- measuring a plant ------------------------------------------------------------------
YOLO_CAN_MISS = False          # YOUR RULE: in simulation YOLO always finds the plant (True = it misses plants that
                               #   span fewer than about YOLO_MIN_PLANT_PX pixels of the D455 RGB image)
YOLO_MIN_PLANT_PX = 32         # PLACEHOLDER (only if YOLO_CAN_MISS)
SHOT_S = 3.0                   # PLACEHOLDER: one shot = trigger, RGB + YOLO + depth + NIR frames, save
PLANT_SPOT_SD_M = 0.08         # PLACEHOLDER: a plant's centre lies this far (SD) from its nominal spot along the row
ARRIVAL_OFFSET_M = 0.20        # PLACEHOLDER: the robot stops this far past a plant's nominal spot ...
ARRIVAL_JITTER_M = 0.10        # PLACEHOLDER: ... give or take this much
PREVIEW_NOISE = 0.03           # PLACEHOLDER: error of the live quality readouts (fill rate, clean-pixel share) when
                               #   standing still ...
PREVIEW_NOISE_MOVING = 0.06    # PLACEHOLDER: ... and when read from frames taken while driving up to the plant
SHOOT_OFFSETS_M = (-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3)
                               # YOUR RULE: the robot may move a little back or forward before shooting - the RL picks
                               #   one of these positions (m from where it stopped) for every plant it measures
NIR_RESHOOT_BELOW = 0.70       # PLACEHOLDER: the hand-written rules take a 2nd (careful) shot from a better NIR spot if
                               #   the first NIR image had less than this share of clean leaf pixels (the RL decides)
ROW_SPACING_ACROSS_M = 0.30    # DATA (rough, Jan 31 flight): plant rows ~0.3 m apart across a bed; the plant measured
                               #   at a spot is in the row nearest the cameras - up to half this to the side of them,
                               #   which moving along the row cannot change
BED_CANOPY_WIDTH_M = 1.4       # DATA: the onions cover ~1.4 m across a bed in April (1.38 m Field 1, 1.40 m Field 2);
                               #   for the cameras to see every row, their view across the bed must be this wide
# the scan light is always on (LIGHT_POWER_W), except during bug-zapper stops

# ---- field and sampling protocol -------------------------------------------------------
FIELD = "f1_05"                # which field (see FIELDS in the docstring): a real map, a pool, or "sq15"
FIELD_ACRES = 15.0             # synthetic fields only (a real field's size comes from its GPS outline)
ROW_LENGTH_M = 247.0           # synthetic fields only: square field (247 m x ~246 m = 15 acres)
BED_SPACING_M = 1.83           # DATA: Vidalia onion beds (Field 1: 1.92 m, Field 2: 1.82 m typical - real fields use their own detected beds)
PASS_EVERY_N_BEDS = 16         # PLACEHOLDER: the robot drives one pass every 16 beds (~30 m apart)
BLOCK_FT = 10.0                # LAB: a sample every 10 ft - at least one plant in every 10-ft block
PLANTS_PER_BLOCK = 6           # PLACEHOLDER: plant spots in a 10-ft block the robot can measure one by one (0.508 m apart)
POINTS_PER_30_ACRES = 50       # LAB: "at least 50 points in a 30-acre field" -> this many parts per 30
                               #   acres (25 parts for 15 acres); each part gets its own estimate
PART_STRESS_FRAC = 0.10        # PLACEHOLDER: a part "needs attention" if at least this share of it is stressed
BASE_TO_FIELD_M = 20.0         # PLACEHOLDER: base/charging spot to the field corner (start of pass 0)
HEALTHY_NDVI = 0.75            # PLACEHOLDER: typical healthy NDVI of your crop
NDVI_STRESS_THRESHOLD = 0.60   # PLACEHOLDER: below this a plant counts as stressed
HEIGHT_MEAN_CM = 35.0          # PLACEHOLDER (synthetic fields; real fields use STAGE_HEIGHT_CM of their flight date)

# ---- real fields: drone map -> what the robot measures (amiga_scout.drone) ---------------
DRONE_NDVI_GAIN = 1.0          # PLACEHOLDER: robot NDVI difference per drone leaf-NDVI difference. Measure it: robot
                               #   and drone on the same plants (e.g. the 50 sample points of each field)
HEIGHT_COVER_EXPONENT = 0.5    # PLACEHOLDER: height ~ canopy cover ^ 0.5 (leaf area grows ~ with height squared)
STAGE_HEIGHT_CM = {"01": 20.0, "02": 25.0, "03": 35.0, "04": 40.0, "05": 50.0, "06": 55.0}
FLIGHT_DATES = {"01": "Jan 31", "02": "Feb 14", "03": "Mar 7", "04": "Mar 18", "05": "Apr 4", "06": "Apr 17"}  # 2024
                               # PLACEHOLDER: typical onion leaf height at each flight date (Jan 31 .. Apr 17)
BARE_COVER_FRAC = 0.25         # DATA: a 10-ft block with less canopy than this share of the field's median is not
                               #   crop (field margin, road, wet spot) - the pass starts/ends where the crop does
GAP_COVER_FRAC = 0.10          # DATA: a spot with less canopy than this share of the median is a stand gap: the robot
                               #   finds only a small, weak plant there

# ---- what the robot is judged AND trained on (YOUR priorities) -------------------------
#   SCORE = ACCURACY (0-100) - HOUR_VALUE_POINTS x mission hours
# Measuring every plant carefully would be most precise but takes far too long; the point of the RL
# brain is to get most of that precision in much less time. This line sets the exchange rate:
HOUR_VALUE_POINTS = 2.0        # YOUR PRIORITY: accuracy points one hour of robot time is worth
                               #   (higher = save time more aggressively, lower = buy more precision)
SCORE_W_BLOCKS = 0.25          # ACCURACY part: the 10-ft sample values (block mean NDVI and height right)
SCORE_W_DETECTION = 0.30       # ACCURACY part: stressed plants found (each miss / half per false alarm costs)
SCORE_W_PARTS = 0.25           # ACCURACY part: per part, flagged right and mean NDVI right
SCORE_W_PROTOCOL = 0.20        # ACCURACY part: share of 10-ft blocks with a plant measured (should be 100%)
BLOCK_TOL_NDVI = 0.08           # a 10-ft block's NDVI off by this much (or more) scores 0 for that block ...
BLOCK_TOL_CM = 5.0              # ... and its height off by this much; each block is scored on its own and averaged
PART_TOL_NDVI = 0.05            # a part's mean NDVI off by this much scores 0 for that part
# a mission that ends stranded scores 0, whatever else happened
# =====================================================================================


@dataclass
class Cfg:
    # power
    pack_voltage_v: float = PACK_VOLTAGE_V
    pack_capacity_ah: float = PACK_CAPACITY_AH
    packs_onboard: int = PACKS_ONBOARD
    spare_pack_sets: int = SPARE_PACK_SETS
    usable_capacity_frac: float = USABLE_CAPACITY_FRAC
    return_trigger_pct: float = RETURN_TRIGGER_PCT
    arrive_min_pct: float = ARRIVE_MIN_PCT
    drive_power_w: float = DRIVE_POWER_W
    hotel_power_w: float = HOTEL_POWER_W
    light_power_w: float = LIGHT_POWER_W
    # bug zapper
    zapper_enabled: bool = ZAPPER_ENABLED
    zap_every_min: float = ZAP_EVERY_MIN
    zap_duration_min: float = ZAP_DURATION_MIN
    zapper_power_w: float = ZAPPER_POWER_W
    # motion
    max_speed_mps: float = MAX_SPEED_MPS
    row_speed_mps: float = ROW_SPEED_MPS          # between stops in a bed (capped at max_speed_mps)
    travel_speed_mps: float = TRAVEL_SPEED_MPS    # driving without stopping (capped at max_speed_mps)
    hop_overhead_s: float = HOP_OVERHEAD_S
    stop_settle_s: float = STOP_SETTLE_S
    nudge_step_m: float = NUDGE_STEP_M
    nudge_speed_mps: float = NUDGE_SPEED_MPS
    row_turn_s: float = ROW_TURN_S
    headland_m: float = HEADLAND_M
    swap_time_s: float = SWAP_TIME_S
    budget_margin: float = 0.15           # extra plants must leave the protocol plan this much energy to spare
    # the cameras and the scene
    camera_height_m: float = CAMERA_HEIGHT_M
    depth_resolution: str = DEPTH_RESOLUTION
    depth_subpixel_rms: float = DEPTH_SUBPIXEL_RMS
    depth_systematic_mm: float = DEPTH_SYSTEMATIC_MM
    depth_fill_field: float = DEPTH_FILL_FIELD
    stereo_min_feature_px: float = STEREO_MIN_FEATURE_PX
    occlusion_half_angle_deg: float = OCCLUSION_HALF_ANGLE_DEG
    nir_lens_focal_mm: float = NIR_LENS_FOCAL_MM
    nir_lens_fnumber: float = NIR_LENS_FNUMBER
    nir_focus_m: float = NIR_FOCUS_M
    nir_exposure_ms: float = NIR_EXPOSURE_MS
    frame_rate_hz: float = FRAME_RATE_HZ
    careful_average_s: float = CAREFUL_AVERAGE_S
    leaf_width_mm: float = LEAF_WIDTH_MM_DEFAULT     # real fields: set from the flight date (LEAF_WIDTH_MM)
    leaf_width_cv: float = LEAF_WIDTH_CV
    leaf_tip_taper_cm: float = LEAF_TIP_TAPER_CM
    canopy_top_sd_cm: float = CANOPY_TOP_SD_CM
    leaf_sway_cm: float = LEAF_SWAY_CM
    leaf_sway_speed_mps: float = LEAF_SWAY_SPEED_MPS
    sway_corr_s: float = SWAY_CORR_S
    wind: float = 1.0                                # leaf sway and speed multiplier (the wind stress test: 3)
    soil_ndvi: float = SOIL_NDVI
    ndvi_view_sd: float = NDVI_VIEW_SD
    view_corr_m: float = VIEW_CORR_M
    glare_max_share: float = GLARE_MAX_SHARE
    glare_ndvi_drop: float = GLARE_NDVI_DROP
    lamp_offset_m: float = LAMP_OFFSET_M
    glare_width_m: float = GLARE_WIDTH_M
    ndvi_calibration_sd: float = NDVI_CALIBRATION_SD
    # measuring a plant
    yolo_can_miss: bool = YOLO_CAN_MISS
    yolo_min_plant_px: float = YOLO_MIN_PLANT_PX
    shot_s: float = SHOT_S
    plant_spot_sd_m: float = PLANT_SPOT_SD_M
    arrival_offset_m: float = ARRIVAL_OFFSET_M
    arrival_jitter_m: float = ARRIVAL_JITTER_M
    preview_noise: float = PREVIEW_NOISE
    preview_noise_moving: float = PREVIEW_NOISE_MOVING
    nir_reshoot_below: float = NIR_RESHOOT_BELOW
    row_spacing_across_m: float = ROW_SPACING_ACROSS_M
    careful_prior_nudges: float = 3.0     # the robot's starting guess of 0.1-m moves per careful plant ... (it then
    careful_prior_shots: float = 1.5      # ... plans with its own running average; see _end_plant)
    quick_prior_nudges: float = 0.5       # the same for quick plants: re-shots when the plant was out of the
    quick_prior_shots: float = 1.3        # ... NIR view, moves the RL chooses
    # field and sampling protocol
    field: str = FIELD                    # real map ("f1_05"), random piece ("f1_05@10"), pool ("f1 f1@5"), square ("sq15")
    route_start_bed: int = -1             # real fields: bed of the first pass (0..PASS_EVERY_N_BEDS-1); -1 = random
    route_corner: int = -1                # real fields: corner the route starts from (0-3); -1 = random
    field_acres: float = FIELD_ACRES      # synthetic fields (real fields: set from the map)
    row_length_m: float = ROW_LENGTH_M    # synthetic fields
    bed_spacing_m: float = BED_SPACING_M  # synthetic fields (real fields: measured)
    pass_every_n_beds: int = PASS_EVERY_N_BEDS
    block_ft: float = BLOCK_FT
    plants_per_block: int = PLANTS_PER_BLOCK
    points_per_30_acres: int = POINTS_PER_30_ACRES
    part_stress_frac: float = PART_STRESS_FRAC
    base_to_field_m: float = BASE_TO_FIELD_M
    # field truth generator (what the robot does NOT know)
    healthy_ndvi: float = HEALTHY_NDVI
    ndvi_stress_threshold: float = NDVI_STRESS_THRESHOLD
    height_mean_cm: float = HEIGHT_MEAN_CM
    ndvi_plant_sd: float = 0.04           # plant-to-plant NDVI scatter (independent per plant)
    height_plant_sd_cm: float = 4.0
    trend_ndvi_sd: float = 0.03           # slow field-wide trend (soil, water)
    trend_wavelength_m: float = 150.0
    patches_per_acre: float = 0.3         # stress patches (pest/disease foci)
    patch_radius_m_min: float = 6.0
    patch_radius_m_max: float = 30.0
    patch_ndvi_drop_min: float = 0.15
    patch_ndvi_drop_max: float = 0.45
    spot_stress_prob: float = 0.0         # isolated single stressed plants (robustness test)
    stress_height_drop_frac: float = 0.25
    drone_ndvi_gain: float = DRONE_NDVI_GAIN          # real fields: drone -> robot (see the constants)
    height_cover_exponent: float = HEIGHT_COVER_EXPONENT
    # robot's map model (what the robot BELIEVES before it sees anything)
    prior_ndvi: float = HEALTHY_NDVI
    prior_height_cm: float = HEIGHT_MEAN_CM
    map_ndvi_spatial_sd: float = 0.12     # how much NDVI varies in neighbour-shared patches
    map_ndvi_plant_sd: float = 0.05       # how much one plant can differ from its neighbours
    map_height_spatial_sd_cm: float = 5.0
    map_height_plant_sd_cm: float = 4.5
    map_length_m: float = 12.0            # how far one measured plant informs its neighbours (metres)
    trust_adapt_rate: float = 0.03        # TRUST: how fast the robot learns the map is off
    trust_max: float = 9.0
    anomaly_z2: float = 4.0               # ANOMALY: a reading (z^2) this far off its prediction loosens that plant
    anomaly_max: float = 25.0             #          from its neighbours (variance multiplier cap)
    # what the robot is judged and trained on: SCORE = ACCURACY - hour_value_points x hours
    hour_value_points: float = HOUR_VALUE_POINTS
    stranded_penalty_points: float = 100.0   # should never happen - the rules prevent it
    max_steps: int = 100000                  # decisions (one per block + one or two per plant) - a safety cap only


# FIELD TYPES: on a real field, the stress is ADDED to the real map ("mixed" field: real variation + known problems)
SCENARIOS = {
    "patches": {},                                                         # stress patches (pest / disease foci)
    "spots": dict(patches_per_acre=0.08, spot_stress_prob=0.03),           # stress NOT shared by neighbours
    "poorer": dict(healthy_ndvi=0.67),                                     # field poorer than the robot assumed
    "as_is": dict(patches_per_acre=0.0),                                   # the real map only, nothing added
}
DEFAULT_SCENARIO = "patches"
# real-world changes used only to TEST a trained model (the training ranges of --randomize stop short
# of these), to see whether it still works when the field day is not like the simulator's defaults
PERTURBATIONS = {
    "soft soil: 25% slower, 50% more motor power": lambda c: dict(
        row_speed_mps=c.row_speed_mps * 0.75, travel_speed_mps=c.travel_speed_mps * 0.75, hop_overhead_s=c.hop_overhead_s * 1.6,
        drive_power_w=c.drive_power_w * 1.5),
    "aged packs: 70% usable": lambda c: dict(usable_capacity_frac=0.70),
    "wind: leaves sway 3x, stops 2.5x less exact": lambda c: dict(
        wind=c.wind * 3.0, arrival_jitter_m=c.arrival_jitter_m * 2.5, plant_spot_sd_m=c.plant_spot_sd_m * 1.5),
    "zapper runs 20 min instead of 15": lambda c: dict(zap_duration_min=20.0),
}

# the train / test split of the real fields: a SPATIAL 80/20 split of BOTH fields (amiga_scout.fields.split_regions).
# In each field one compact end holding 20% of its crop is the TEST region (never trained on, never used to pick a
# model); the rest, minus a buffer strip next to the test region, is the TRAINING region. A buffer is standard in
# spatial validation: neighbouring plants look alike, so a test region right next to the training area would flatter
# the model (Roberts et al. 2017). The regions are the same for every flight of a field (one bed grid).
SPLIT_TEST_SHARE = 0.20        # share of each field's crop area held out for testing
SPLIT_BUFFER_M = 30.0          # strip left out between the two regions (about one pass apart; stress patches are 6-30 m)
# real training regions (5-21.5 acres: whole, and pieces) and BIG fields made of real drone data too: the training
# regions laid out 2-3 times side by side (fields.tiled_field), cut into 25-40-acre pieces - the whole real fields
# are 24-30 acres, and the test regions must never be trained on. Each real token expands to its 6 flights, so a
# third of the missions are big.
TRAIN_FIELDS = ("f1~train f2~train f1~train@5 f1~train@10 f1~train@15 f2~train@5 f2~train@10 f2~train@15 "
                "f1~train*2@25 f1~train*2@35 f2~train*3@30 f2~train*3@40")
TEST_FIELD = "f1~test f2~test"      # the two test regions, every flight: never used in training or model selection
SELECT_FIELDS = ("f1~train f2~train", "f1~train@5 f1~train@10 f2~train@5 f2~train@10",
                 "f1~train*2@30 f2~train*3@35")        # picking a model (new seeds): medium, small and big fields
BIG_TEST_FIELDS = "f1~test*5 f2~test*3x2"          # ~30- and ~28-acre fields made only of the test regions (held out)
