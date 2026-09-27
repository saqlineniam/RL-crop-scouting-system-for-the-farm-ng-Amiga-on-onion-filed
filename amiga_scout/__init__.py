r"""
amiga_scout - RL brain for the farm-ng Amiga scouting REAL Vidalia onion fields (and synthetic ones)
=================================================================================================

WHAT THIS SIMULATES
    One scouting mission of the farm-ng Amiga on one field, exactly as the lab protocol says:

        TAKE A SAMPLE EVERY 10 FT: at least one plant of every 10-ft block is measured
        (RGB + YOLO finds the plant, D455 depth -> height, NIR camera -> NDVI). Enforced by code.

    THE FIELDS ARE REAL. Two commercial Vidalia onion fields (Vidalia, GA, 2024) were mapped by a DJI
    Mavic 3M on six dates (Jan 31 - Apr 17); amiga_scout.drone turns every flight into what the robot
    would see (see its docstring):
        Field 1  30.0 acres, 307 beds ~1.92 m apart
        Field 2  23.7 acres, 104 beds ~1.82 m apart
    Both are split 80/20 in space (amiga_scout.fields.split_regions): in each, one compact end with 20% of the
    crop is the TEST region the trained robot never sees before `validate`; the rest, minus a 30-m buffer strip,
    is the TRAINING region.
    A field is used whole, with its real outline. The field is GPS-mapped in advance, so the robot knows
    the outline and the beds; its route is fixed: it drives ALONG THE BEDS, one pass every
    PASS_EVERY_N_BEDS beds (~30 m), snaking across the field, and every pass is cut into 10-ft blocks of
    6 plant spots (0.508 m apart). Which bed the first pass takes and which corner it starts from vary
    from mission to mission (more route variety from the same maps).
    Synthetic square fields (the earlier simulator's generator, "sq5" .. "sq15") remain available.

    WHAT IS TRUE AT EACH PLANT (hidden from the robot):
        NDVI    = the robot's healthy level + (drone leaf NDVI of this spot - the field's median) x gain
                  + the plant's own deviation - any added stress
        height  = typical height for the growth stage x (spot canopy cover / median) ^ 0.5
                  + the plant's own deviation, less where stressed
      Drone -> robot: the robot looks at leaves (not the sandy soil), from close range, with its own light,
      so only the drone's plant pixels are used and only RELATIVE differences are carried over (the robot's
      NDVI scale differs from the drone's: DRONE_NDVI_GAIN, PLACEHOLDER until measured side by side). The
      drone DEMs are too coarse for onion leaves, so height comes from canopy size and a stage height
      (STAGE_HEIGHT_CM, PLACEHOLDER).

    FIELD TYPES (what problems the field has on the day). The real fields look healthy, so problems are
    ADDED on top of the real maps - "mixed" fields: real variation + known stress with exact ground truth.
        as_is     the real map only (natural variation, pivot tracks, weak zones)
        patches   + stress patches (pest / disease foci, 6-30 m)
        spots     + isolated stressed plants (not shared by neighbours)
        poorer    the whole field is poorer than the robot assumes (its prior is off)

    THE CAMERAS ARE MODELLED AS INSTRUMENTS (amiga_scout/sensors; sources in docs/references.md): the Intel
    RealSense D455 gives height - no depth closer than its minimum distance, noise growing with distance squared,
    thin leaf tips lost, neighbours hiding part of the top off-centre - and a red + NIR camera (stand-in: JAI
    AD-130GE) gives NDVI - pixel size, focus and motion blur vs the thin onion leaves, glare from the scan light,
    leaf-angle errors tied to the viewpoint. `python -m amiga_scout cameras` shows the curves.

    The robot decides at two levels (HAM-PPO's per-unit choice), from the live camera view and its own map:
      AT EACH BLOCK
        level 1  QUICK    -> each plant shot once, or
                 CAREFUL  -> frames averaged for CAREFUL_AVERAGE_S, a second shot from another spot allowed
        level 2  how many of the block's plants: 1 (the protocol minimum) .. all 6
      AT EACH OF THOSE PLANTS: WHERE TO TAKE THE PICTURE
        The robot stops ~20 cm short of the plant's mapped spot (the map is good to ~8 cm). Driving in, it saw
        live previews (depth fill rate, clean NIR leaf share) from the positions it passed. It then picks
        SHOOT at -30 .. +30 cm (10-cm moves, each costing time and energy) or DONE (once it has an NDVI
        reading). Two shots without NDVI -> it centres on the YOLO box; six -> it takes the next plant.
        Cameras are mounted at CAMERA_HEIGHT_M (1.5 m, from `mount`) with the wide side across the bed;
        glare, blur and leaf angle still depend on where the robot stops, so position matters.
    In simulation YOLO always finds the plant (YOLO_CAN_MISS = False).

    It is judged on the accuracy of its final map, minus the time it took:
        - every 10-ft block (the lab's sample values: mean NDVI and height)
        - every plant on the route (stressed or not)
        - every part of the field (the lab's "at least 50 points per 30 acres")
    The robot has NO drone map: it builds its own map as it goes (Gaussian-process map with TRUST and
    ANOMALY, as before).

    Batteries (automatic, the lab procedure): at the return level the robot finishes the plant, drives
    straight on to the end of the pass, follows the headland home, a person swaps the packs
    (SPARE_PACK_SETS times at most) and it comes back to the same plant. No working-day limit. Extra
    plants are only allowed while the energy left still covers the protocol plant of every block ahead.
    Every 30 min of field work: a 15-min bug-zapper stop (scan light off, the zapper has its own).

    Decisions use a two-level conditional action tree with masks at both levels (Khosravi et al. 2025,
    HAM-PPO). Hand-written rules to beat include the Gaussian-process uncertainty rule (Kumar et al. 2019).

    ACCURACY is scored per 10-ft block, per plant and per part, then averaged, so a measurement counts as soon
    as it is made (blocks 25%, stressed plants found 30%, parts 25%, every block sampled 20%).

HOW THE POLICY IS TRAINED  (`train`, amiga_scout.rollout)
    One block decision is worth only ~0.001-0.02 points of a mission's score, far below the measurement noise
    PPO sees when it compares decisions (0.1-0.5 points) - so `train` uses the simulator to compare choices:
    at sampled decisions of training missions, every legal choice is tried in a copy of the mission with the SAME
    measurement noise (common random numbers: PEGASUS, the "vine" of TRPO), valued by the accuracy it adds and
    the hours it costs over the next blocks, and a network learns these values from what the robot knows. The
    policy takes the best choice (policy iteration with rollouts: Bertsekas 2020, Lagoudakis & Parr 2003); 3
    rounds, each adding the moments the newer policy visits. Positioning is learned the same way and kept only
    if it beats the hand-written positioning on held-out missions. HAM-PPO stays available: `--method ppo`.

FIELDS (`--field`, space-separated pool; one is drawn per mission)
    f1_05        Field 1, flight 05 (Apr 4), whole      f1           all six Field 1 flights, whole
    f1_05~test   its TEST region (20%, held out)        f1~train@10  10-acre pieces of Field 1's training region
    f1_05~train  its TRAINING region                    sq10         a synthetic 10-acre square (old generator)
    Training uses both fields' training regions, whole and in random 5/10/15-acre pieces (a new piece every
    mission: random place, random shape 1:2-2:1). The test regions are only for `validate` / `fieldwise`.

HOW TO VALIDATE A TRAINED MODEL  (`validate`)
    The held-out test regions of both fields (all six dates, all four field types), paired with the
    hand-written rules on the same missions (95% CIs, win rate); hard safety and protocol checks; both fields
    whole (big-field behaviour; 80% of their area was trained on); stress tests (soft soil, aged packs, wind,
    longer zapper). Ends with a READY / NOT READY verdict.

HOW TO RUN (with the .venv activated, from the repository folder SIM/onion_robot; see README.md)
    python -m amiga_scout extract-all                 # drone survey -> robot-view maps
    python -m amiga_scout describe --field f1_05~test
    python -m amiga_scout compare --field "f1~test f2~test" --episodes 6 --brief
    python -m amiga_scout train --randomize                        # -> models/onion_rl.pt
    python -m amiga_scout curve --model-path models/onion_rl       # the best round
    python -m amiga_scout validate --models models/checkpoints/onion_rl_it3.pt
    Any setting can be overridden without editing:  --set name=value  (repeatable)

VALUE TAGS BELOW
    SPEC         from the farm-ng Amiga documentation / store page
    LAB          from the lab (sampling protocol, runtime, zapper, professor's estimate)
    DATA         measured from the Vidalia onion drone survey (amiga_scout.drone)
    YOUR RULE    something you asked for
    PLACEHOLDER  a guess - measure it in the lab and replace it
    REAL         from your D455 log (depth quality vs camera distance)

PACKAGE LAYOUT
    config.py     every setting (lab values on top), field types, stress tests, train/test split
    fields.py     real maps, 80/20 regions, pieces, squares   actions.py   block choices + where to shoot
    env/          the mission: route, energy, truth, measuring, mapping, observation -> mission.py
    policies/     hand-written rules, the learned policy (qnet), PPO models, readable trees
    rollout.py    the training method   training.py  `train` (+ HAM-PPO)    evaluation.py running missions
    commands/     CLI commands
    drone/        the drone survey -> robot-view maps (beds, extraction, analysis)
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")   # small matrices: one thread per process is fastest, and many
                                     # processes (training, validate) must not fight over the cores
