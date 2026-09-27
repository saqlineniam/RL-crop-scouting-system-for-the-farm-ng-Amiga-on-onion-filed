"""`describe`: the field, route, batteries and time vs precision for one mission."""
from __future__ import annotations

from dataclasses import fields

from ..config import Cfg, DEFAULT_SCENARIO, FLIGHT_DATES, PROFESSOR_ACRES_PER_CHARGE
from ..env import AmigaMissionEnv
from ..evaluation import build_cfg
from ..fields import field_key, field_map, load_field, parse_name, split_regions, tile_of
from ..sensors import D455_DEPTH, min_z


def cmd_describe(args):
    env = AmigaMissionEnv(build_cfg(args))
    env.reset(seed=args.seed)
    cfg = env.cfg
    lengths = (env.c1 - env.c0 + 1) * env.dxp
    if env.real:
        f = load_field(cfg.field)
        if tile_of(cfg.field) != (1, 1):
            t = field_map(cfg.field)
            print(f"       TILED: its region laid out {t['tiles'][0] * t['tiles'][1]} times ({t['tiles'][0]} across the beds x "
                  f"{t['tiles'][1]} along them, every other copy mirrored) = {t['acres']:.1f} acres of real drone data")
        print(f"Field: {cfg.field} = REAL map ({FLIGHT_DATES.get(f['date'], f['date'])} 2024), {cfg.field_acres:.1f} acres of crop, beds "
              f"{f['bed_spacing_m']:.2f} m apart; growth-stage height {cfg.height_mean_cm:.0f} cm")
        _, region, size = parse_name(cfg.field)
        if region:
            sp = split_regions(field_key(cfg.field))
            cut = "a band of beds" if sp["axis"] == 0 else "the same stretch of every bed"
            print(f"       its {region.upper()} region (80/20 spatial split, cut as {cut}): test = {sp['test_share']:.0%} of "
                  f"the crop, training = {sp['train_share']:.0%}, with a {sp['buffer_cells']}-cell buffer strip between")
        if env.piece:
            k0, k1, j0, j1 = env.piece
            what = (f"a random PIECE of {'its ' + region + ' region' if region else 'the field'} (new every mission)"
                    if size else f"the {region} region")
            print(f"       {what}: beds {k0}-{k1 - 1} ({k1 - k0} beds) x {(j1 - j0) * f['step_m']:.0f} m along them "
                  f"(spots {j0}-{j1 - 1}); the whole field is {f['acres']:.1f} acres")
        print(f"Route (mission seed {args.seed}): first pass on bed {cfg.route_start_bed}, from corner {cfg.route_corner}; "
              f"a pass every {cfg.pass_every_n_beds} beds ({env.dy:.1f} m apart) -> {env.R} passes of "
              f"{lengths.min():.0f}-{lengths.max():.0f} m (the real outline)")
    else:
        print(f"Field: {cfg.field} = synthetic square, {cfg.field_acres:.0f} acres = {cfg.row_length_m:.0f} m rows x "
              f"{env.width_m:.0f} m wide")
        print(f"Route: a pass every {cfg.pass_every_n_beds} beds ({env.dy:.1f} m apart) -> {env.R} passes of "
              f"{lengths.max():.0f} m")
    print(f"       {env.NBv} blocks of {cfg.block_ft:.0f} ft, {env.K} plant spots each -> {env.Nv} plants on the route")
    print(f"Protocol: at least 1 plant measured in every {cfg.block_ft:.0f}-ft block ({env.NBv} samples at least); "
          f"{env.Z} parts for {cfg.field_acres:.1f} acres (lab: {cfg.points_per_30_acres} per 30 acres)")
    print(f"Truth on this mission (field type '{DEFAULT_SCENARIO}'): "
          f"{env.n_stressed} stressed plants ({env.n_stressed / env.Nv:.1%}), "
          f"{int(env.z_flag_true[:env.Z].sum())} parts need attention"
          + (f", {int(env.gap.sum())} stand gaps" if env.real else ""))
    zap = (f"on: {cfg.zap_duration_min:.0f} min stop every {cfg.zap_every_min:.0f} min of work "
           f"(+{env.zap_share:.0%} time)" if cfg.zapper_enabled else "off")
    zmin = min_z(D455_DEPTH, cfg.depth_resolution)
    print(f"Cameras: {cfg.camera_height_m:.2f} m up, looking down. D455 depth at {cfg.depth_resolution} (no depth closer "
          f"than {zmin:.2f} m: plants up to {100 * (cfg.camera_height_m - zmin):.0f} cm); NIR: JAI AD-130GE, "
          f"{cfg.nir_lens_focal_mm:g} mm f/{cfg.nir_lens_fnumber:g}; leaves {cfg.leaf_width_mm:g} mm wide at this stage "
          f"(`cameras` shows the curves and per-shot errors)")
    print(f"Speed: {env.v_row:.2f} m/s ({env.v_row / 0.44704:.1f} mph) between stops in a bed, {env.v_travel:.2f} m/s "
          f"({env.v_travel / 0.44704:.1f} mph) driving without stopping; top speed {cfg.max_speed_mps / 0.44704:.1f} mph")
    print(f"Power: drive {cfg.drive_power_w:.0f} W + hotel {cfg.hotel_power_w:.0f} W + scan light {cfg.light_power_w:.0f} W "
          f"(always on, off during zapper stops) -> {env.capacity_wh / env._drive_power():.1f} h of driving per charge; "
          f"bug zapper {zap}")
    usable = env.capacity_wh - env.reserve_wh
    print(f"Battery: both packs = {env.capacity_wh:.0f} Wh, {usable:.0f} Wh before the {cfg.return_trigger_pct:.0f}% "
          f"return rule; {cfg.spare_pack_sets} spare pair(s) -> {usable * (1 + cfg.spare_pack_sets):.0f} Wh per mission; "
          f"no working-day limit")
    tq, eq = env.block_cost("quick", 1)
    tc, ec = env.block_cost("careful", 1)
    print(f"Per 10-ft block incl. driving on and the zapper's share (planning estimate; a careful plant uses the robot's "
          f"starting guess of {cfg.careful_prior_nudges:g} moves and {cfg.careful_prior_shots:g} shots, then its own "
          f"average): 1 quick plant {tq:.0f} s / {eq:.2f} Wh, 1 careful plant {tc:.0f} s / {ec:.2f} Wh")
    wall, wh = env.protocol_plan(from_start=True)
    print(f"\nMinimum protocol on this field (1 quick plant per block): ~{wall / 3600:.1f} h, ~{wh:.0f} Wh = "
          f"{wh / usable:.2f} charges of both packs")
    print(f"For reference: at the protocol minimum this model gets ~{usable / (wh / cfg.field_acres):.0f} acres out of "
          f"both packs WITH our load and stops; the professor's rough estimate was ~{PROFESSOR_ACRES_PER_CHARGE:.0f} acres "
          f"with NO load. Measure the real draw to pin this down.")
    print("\nTime vs precision - the same number of plants in every block (planning estimate):")
    print(f"  {'plants/block':>12s} {'plants':>7s} {'how':>8s} {'hours':>6s} {'charges':>8s}  "
          f"batteries enough (1 + {cfg.spare_pack_sets} spare)?")
    for k in range(1, env.K + 1):
        for mode in ("quick", "careful"):
            t, e = env.block_cost(mode, k)
            w, h = wall + env.NBv * (t - tq), wh + env.NBv * (e - eq)
            print(f"  {k:12d} {k * env.NBv:7d} {mode:>8s} {w / 3600:6.1f} {h / usable:8.2f}  "
                  f"{'yes' if h <= usable * (1 + cfg.spare_pack_sets) else 'no'}")
    print(f"\nSCORE = ACCURACY (0-100) - {cfg.hour_value_points:g} x hours  (HOUR_VALUE_POINTS: what one hour of robot time "
          f"is worth to you).\nThe robot is trained on exactly this: spend extra time only where it buys enough accuracy.")
    print("\nAll settings (override with --set name=value):")
    for f in fields(Cfg):
        print(f"  {f.name:28s} {getattr(cfg, f.name)}")
