# Learning Where to Measure in Onion Fields with a farm-ng Amiga

**Simulation-based reinforcement learning for adaptive crop scouting on real drone-mapped Vidalia onion fields.**

*Keywords: reinforcement learning · agricultural robotics · crop scouting · precision agriculture · farm-ng Amiga · onion (Allium cepa) · NDVI · plant phenotyping · Intel RealSense D455 · drone mapping (DJI Mavic 3M) · active sensing · simulation*

A farm-ng Amiga robot drives a fixed route along onion beds and measures plants: **height** with an Intel RealSense D455, and **NDVI** with a red + NIR camera. A lab protocol requires at least one plant in every 10-ft block. Beyond that, every extra plant costs time.

This project learns **how much to measure where**. At every block the policy picks quick or careful and how many of the block's 6 plants (1–6). It trades map accuracy against robot time:

> **SCORE = ACCURACY (0–100) − λ × mission hours**   (λ = value of one robot hour; λ = 1 here)

The fields are **real**: two commercial onion fields (30 and 23.7 acres), mapped by a DJI Mavic 3M on six dates in 2024 and turned into what the robot would see. The policy is trained by **simulation-based policy iteration with common random numbers**, after on-policy PPO (HAM-PPO) failed on this problem.

---

## Results at a glance

Best model: `onion_rl3_it3`, λ = 1. All test fields are held out: they were never used in training or to pick the model. The ± values are 95% confidence intervals of paired differences (same fields, routes and stress for every strategy).

| Test | RL vs the best hand-written rule (score) | Best rule there | Time: best rule | Time: RL |
|---|---|---|---|---|
| Small held-out fields (5–6 acres), all 4 field types | **tie** (−0.3 to 0.0) | every plant, quick | 3.8 h | **3.5–3.8 h** |
| **Big held-out fields (~30 acres, never seen)**, `validate` | **+1.1 ± 0.8**, wins 83% of missions | minimum (1 plant per block) | 5.5 h | 8.3 h |
| Same big fields, 576 missions, `fieldwise` | **+2.4 ± 0.4**, better on all 12 flight dates, wins 41 of 48 date × type cases | minimum | 5.5 h | 9.3 h |
| Whole real fields (24–30 acres, 80% of area seen in training) | **+2.8 ± 1.8**, wins 92% | minimum | 5.0 h | 7.2 h |
| Stress tests (soft soil, aged packs, wind, longer zapper) | not worse in any | every plant, quick | 3.7–4.6 h | **3.4–4.0 h** |
| Safety: >1,300 test missions | **0 stranded, 0 illegal actions, 0 unsampled blocks** | – | – | – |

**Reading the time columns:**
- **Small fields:** the RL matches "every plant" while finishing slightly sooner.
- **Big fields:** the RL spends more time than the minimum rule (about 3 extra hours) and less than "every plant, quick", which takes 13.6 h and needs a battery swap. The extra hours buy more than they cost: +3.9 to +5.0 accuracy points, so the score (accuracy − 1 × hours) comes out ahead.

**Where the RL shines: big fields, where no fixed rule is right.** On a 30-acre field:
- the minimum rule is fast (5.5 h) but misses stress;
- "every plant" is accurate but takes 13.6 h and is limited by the batteries;
- the RL spends **9.3 h**, reaches **74.9** accuracy (every plant: 75.6), and measures more **where the field shows stress or surprise**.

It gets the part-level flags right more often than either rule. On small fields it learns on its own to measure everything, which is the best strategy there. **It is the only strategy that is near the best at every field size.**

### All scores

**Held-out test regions (5–6 acres), 24 missions per field type** (score = accuracy − 1 × hours, mean ± 95% CI):

| Strategy | Patches | Spots | Poorer | As is | Hours |
|---|---|---|---|---|---|
| 1 plant per block, quick (minimum) | 72.9 ± 3.9 | 58.8 ± 2.9 | 62.3 ± 4.4 | 75.2 ± 3.7 | 1.0 |
| 2 plants per block, careful | 73.6 ± 3.4 | 63.8 ± 3.0 | 64.8 ± 4.3 | 76.0 ± 3.0 | 2.6 |
| Every plant, careful | 76.6 ± 2.5 | 79.6 ± 2.4 | 69.2 ± 5.5 | 78.6 ± 2.4 | 7.2 |
| **Every plant, quick** | **81.1 ± 2.3** | **83.2 ± 2.4** | **73.1 ± 4.7** | **82.2 ± 2.1** | 3.8 |
| Adaptive rule (hand tree) | 75.8 ± 3.0 | 66.2 ± 2.3 | 67.7 ± 4.9 | 77.7 ± 2.6 | 2.8–4.6 |
| Uncertainty rule (GP-driven, Kumar et al. 2019) | 77.6 ± 2.2 | 68.6 ± 2.4 | 69.1 ± 5.4 | 79.2 ± 2.3 | 3.2–6.4 |
| Battery-filling rule | 76.6 ± 2.5 | 79.6 ± 2.4 | 69.2 ± 5.5 | 78.6 ± 2.4 | 7.2 |
| **RL (this work)** | **81.1 ± 2.2** | **82.9 ± 2.4** | **73.0 ± 4.8** | **82.1 ± 2.1** | 3.5–3.8 |

**Big fields, mean score over all field types:**

| Strategy | Held-out ~30-acre fields (`validate`) | Whole real fields (24–30 acres) | Accuracy / hours, held-out big fields |
|---|---|---|---|
| Minimum | 69.5 | 68.9 | 75.0 / 5.5 h |
| 2 plants per block, careful | 63.8 | 64.6 | 77.2 / 13.4 h |
| Every plant, careful | 63.7 | 64.6 | 77.2 / 13.5 h |
| Every plant, quick | 66.3 | 66.6 | 79.9 / 13.6 h |
| Adaptive rule | 65.1 | 66.0 | 78.5 / 13.4 h |
| Uncertainty rule | 65.7 | 66.2 | 79.3 / 13.6 h |
| Battery-filling rule | 63.4 | 63.7 | 76.8 / 13.4 h |
| **RL** | **70.5** | **71.7** | **78.8 / 8.3 h** |

**Held-out ~30-acre fields by field type** (576 missions; RL minus the best rule, paired):

| Patches | Poorer | As is | Spots | All types |
|---|---|---|---|---|
| **+1.6 ± 0.5** | **+1.3 ± 0.6** | **+1.2 ± 0.7** | +1.0 ± 1.1 | **+2.4 ± 0.4** |

- Better than the best rule on every one of the 12 flight dates (+1.3 to +4.2).
- Spots on Field 1's later flights is the one weak spot (−1.3 to −1.5, within noise). Isolated stressed plants can only be found by measuring them.

**Score breakdown on the held-out big fields** (accuracy parts, 0–100):

| Strategy | Score | Accuracy | Hours | Blocks | Stress found | Parts | Plants/block |
|---|---|---|---|---|---|---|---|
| Minimum | 63.2 | 68.7 | 5.5 | 59.2 | 58.2 | 65.8 | 1.00 |
| Every plant, quick | 62.1 | 75.6 | 13.6 | 68.8 | 71.8 | 67.6 | 3.46 (battery-limited) |
| **RL** | **65.6** | 74.9 | 9.3 | 65.2 | 69.9 | **70.6** | 2.10 |

**Model selection (training fields, new routes and stress), by training round:**

| Round | Overall | Whole regions (~20 ac) | 5–10-acre pieces | Big tiled fields (30–35 ac) |
|---|---|---|---|---|
| 0: base rule | 71.62 | 73.10 | 74.05 | 67.71 |
| 1 | 73.36 | 75.06 | 76.66 | 68.36 |
| 2 | 73.51 | 75.20 | 76.31 | 69.01 |
| **3 (chosen)** | **73.90** | **75.60** | **76.89** | **69.19** |
| Best rule | 71.62 (minimum) | 73.10 (minimum) | 76.31 (every plant) | 67.71 (minimum) |

**The value of time.** The same model evaluated at other λ, without retraining:

| λ (points per hour) | Small fields: RL vs best rule | Big fields: RL vs best rule |
|---|---|---|
| 0.5 | −0.2 (tie) | **+1.4** |
| 1 | −0.3 (tie) | **+2.6** |
| 2 | −0.5 (tie) | −0.6 (the minimum is near-optimal) |
| 3 | −0.6 (tie) | −0.2 (the minimum is near-optimal) |

The RL adds the most when time is worth about 1 point per hour or less. At higher λ, measuring the minimum is almost optimal, and the RL converges to it.

### What made the difference

These are the lessons of this project, each backed by an experiment described in *How the policy is trained*:

1. **PPO could not learn this.** One block decision changes a mission score by 0.001–0.02 points; the noise PPO saw when comparing decisions was 0.1–0.5 points; and its value estimate explained less than predicting the average would. HAM-PPO ended up between the rules everywhere.
2. **Comparing choices in copies of the mission with the same measurement noise** (common random numbers) made comparisons 25–650× cleaner. Learning from those comparisons (rollout policy iteration) beat the rules.
3. **An ensemble of 5 networks** stopped random network errors from deciding near-equal choices. A single network picked the truly best option only at chance level.
4. **Big training fields built from real drone data** (tiled training regions) beat synthetic square fields.
5. **Two value scales, blended by field size.** Values in points work best on small fields; values per block's share of the field work best on big fields.
6. **Careful mode only when the networks agree it pays** removed careful choices the data didn't support.
7. **Scoring each block, plant and part on its own** removed flat regions where early measurements were worth nothing.

---

## Data availability

The drone survey, the extracted robot-view maps (`data/*.npz`, `data/geometry.json`) and the trained models are **not included in this repository**. For access to the data, please contact **Saklain Niam** (saqlineniam@gmail.com).

All code is included. With the data in place, `python -m amiga_scout extract-all` rebuilds the maps, and every result above can be reproduced with the commands below. Without the data, the simulator still runs on synthetic fields (`--field sq10`), and the tests that need real maps are skipped.

---

## The machine (as simulated)

| Item | Value |
|---|---|
| Robot | farm-ng Amiga. Top speed 3.6 mph (a ceiling); in the bed 0.89 m/s stop-and-go; headland travel 1.34 m/s |
| Batteries | 2 packs on board + 1 spare pair. At 20% charge the robot drives to the end of the pass, goes home, the packs are swapped, and it resumes at the same plant |
| Bug zapper | a 15-min stop after every 30 min of field work |
| Scan light | always on, except during zapper stops |
| Route | fixed, along the beds, one pass every 16 beds (~30 m); the field is GPS-mapped beforehand |
| Protocol | at least 1 plant measured in every 10-ft block (6 plant spots, 0.508 m apart); parts = 50 per 30 acres. Enforced by action masks; energy masks guarantee the robot is never stranded |
| Height | Intel RealSense D455 depth, 1280×720 |
| NDVI | JAI AD-130GE red + NIR prism camera (stand-in), 2.8 mm f/2 focused at 1.31 m |
| Mount | cameras 1.5 m up, looking down, wide side across the bed (`mount` explains why) |
| Plant finding | D455 RGB + YOLO (never misses in simulation) |
| Robot's map | block-level Gaussian process with TRUST and ANOMALY; the robot never sees the drone map |

**Accuracy** has four parts, each scored per block, plant or part and averaged:
- 10-ft block values (NDVI and height): 25%
- stressed plants found (misses cost 1, false alarms 0.5): 30%
- parts (flag and mean NDVI): 25%
- protocol (every block sampled): 20%

The details of every part of the machine follow below.

## Repository layout

```
onion_robot/
├── amiga_scout/                 the Python package
│   ├── __init__.py              overview of the whole method (python -m amiga_scout --help shows it)
│   ├── __main__.py / cli.py     one command line for everything
│   ├── config.py                ALL settings: lab values on top, per-mission Cfg, field types, stress tests
│   ├── paths.py                 where data, models, figures and the drone survey are
│   ├── actions.py               per block QUICK/CAREFUL x 1-6 plants; per plant SHOOT at -30..+30 cm or DONE
│   ├── fields.py                real maps, their 80/20 train/test regions, pieces, synthetic squares
│   ├── sensors/                 the cameras: datasheet profiles, optics formulas, one shot of one plant
│   ├── mathutil.py              small numeric helpers
│   ├── env/                     the mission, one concern per file
│   │   ├── mission.py           AmigaMissionEnv: reset, step (a block decision or a shot), summary
│   │   ├── route.py             beds, passes on the real outline, travel, trips home
│   │   ├── energy.py            power, zapper, battery swaps, energy plan, action masks
│   │   ├── truth.py             hidden truth: drone map -> robot scale, added stress
│   │   ├── measuring.py         arriving (previews), moving the cameras, shooting a plant
│   │   ├── mapping.py           the robot's own map (Gaussian process) and ACCURACY
│   │   └── observation.py       what the policy sees
│   ├── policies/
│   │   ├── rules.py             hand-written strategies to beat
│   │   ├── qnet.py              the learned policy (value of every choice, from what the robot knows)
│   │   ├── ham.py               hierarchical masked PPO policy (HAM-PPO style, the earlier method)
│   │   ├── model.py             a PPO model as a policy
│   │   └── tree.py              readable decision trees from a trained model
│   ├── evaluation.py            running missions in parallel, result tables
│   ├── rollout.py               training: policy iteration with rollouts and common random numbers
│   ├── training.py              `train` (rollout by default; HAM-PPO with --method ppo)
│   ├── commands/                describe, cameras, mount, compare, validate, fieldwise, curve, watch, explain
│   └── drone/                   drone survey -> what the robot would see
│       ├── io.py                fields, flights, file paths, pixel helpers
│       ├── beds.py              exact bed direction, every furrow, flight registration
│       ├── extract.py           orthomosaic -> bed-aligned robot-view map
│       └── analysis.py          statistics, yield check, DEM check, figures
├── tests/                       pytest: protocol, reward = score, route, bookkeeping, data, cameras, split, rollout
├── docs/references.md          every datasheet and paper behind the design, and what it supports
├── docs/PROJECT_SUMMARY.md      a compact overview of the whole design
├── data/                        robot-view maps and geometry (not included: contact the author)
├── figures/                     figures from `analyse`
├── models/                      trained models and checkpoints (not included)
├── pyproject.toml, requirements.txt
```

## Setup

Python 3.12:
```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt        # Linux/macOS: .venv/bin/pip install -r requirements.txt
```
The raw drone survey path is set by `DRONE_ROOT` in `amiga_scout/paths.py`. The extracted maps go in `data/` (not included; see *Data availability*).

## Commands

Activate the environment, then run everything from the repository folder:

```
python -m amiga_scout extract-all          # every flight -> robot-view map (~2 min each, skips existing)
python -m amiga_scout analyse --dem        # statistics, yield check, DEM check, figure
python -m amiga_scout describe --field f1_05~test
python -m amiga_scout compare --field "f1~test f2~test" --episodes 12 --brief
python -m amiga_scout train --randomize --set hour_value_points=1          # -> models\onion_rl.pt (+ every round)
python -m amiga_scout curve --model-path models\onion_rl --set hour_value_points=1       # pick the best round
python -m amiga_scout validate --models models\checkpoints\onion_rl_it3.pt --set hour_value_points=1   # test regions
python -m amiga_scout fieldwise --models models\checkpoints\onion_rl_it3.pt --set hour_value_points=1  # per flight
python -m amiga_scout watch --field f2_05~test --policy models\onion_rl
python -m amiga_scout explain --policy models\onion_rl
python -m amiga_scout train --method ppo --randomize --warm-start all_plants_quick --steps 6000000   # the earlier method
python -m pytest                            # tests (~1 min)
```
Any setting can be changed without editing code, for example `--set hour_value_points=3`. `python -m amiga_scout describe` lists every setting.

### Field names

| Name | Meaning |
|---|---|
| `f1_05` | Field 1, flight 05 (Apr 4), whole |
| `f1` | All Field 1 flights, whole |
| `f1_05~test` | The **test region** of `f1_05` (20% of the field, never trained on) |
| `f1_05~train` | The **training region** of `f1_05` (the rest, minus a 30-m buffer) |
| `f1~train@10` | A random 10-acre **piece** of the training region of every Field 1 flight: a new piece every mission (random place, random shape 1:2–2:1) |
| `f1_05@10` | A random 10-acre piece anywhere in `f1_05` |
| `f1_05~train*2` | A **big field made only of real drone data**: the training region laid out twice side by side (the copy mirrored, so beds and texture run on across the seam). 43 acres for Field 1 |
| `f1~train*2@30` | A random 30-acre piece of it (every Field 1 flight). Pieces hold the asked amount of crop and keep the real outline |
| `f1~test*5`, `f2~test*3x2` | ~30- and ~28-acre fields made only of the **test regions** (held out): the big-field test. `*3x2` = 3 copies across the beds × 2 along them |
| `sq10` | A synthetic 10-acre square (the old generator; no longer used by default) |
| `"f1~train f2~train"` | A pool: one is drawn per mission |

### Train / test split (80/20, spatial)

Both fields are used for training and for testing. In each field, one compact end holding 20% of the crop is the **test region**; the rest, minus a 30-m buffer strip next to it, is the **training region**. A buffer is standard in spatial validation: neighbouring plants look alike, so a test region right next to the training area would flatter the model (Roberts et al. 2017). The cut goes across each field's longer side so the test region is compact, and of the two ends the one whose plant NDVI and canopy cover are closer to the whole field's is held out. The regions are the same on all six flights (one bed grid per field).

| | Field 1 | Field 2 |
|---|---|---|
| Cut | a band of beds (the field is 589 m across the beds, 393 m along them) | the same stretch of every bed (189 m across, 551 m along) |
| Test region | 6.0 acres (20%), 4 passes | 4.7 acres (20%), 6 passes |
| Buffer | 16 beds (31 m) | 60 spots (30 m) |
| Training region | 21.5 acres (72%) | 17.6 acres (74%) |

- **Training** (`TRAIN_FIELDS`):
  - both training regions whole (17.6 and 21.5 acres);
  - random 5, 10 and 15-acre pieces of them;
  - **big fields of 25–40 acres made of the same real drone data**: the training regions laid out 2–3 times side by side and cut into pieces. A third of the missions are big.
  - Every flight is used.
- **Picking a model** (`curve`): small, medium and big training fields again, with new routes and new stress (missions 80000+).
- **Testing** (`validate`, `fieldwise`), missions 90000+:
  - the two test regions (5–6 acres), all flights;
  - **~30-acre fields made only of the test regions** (`f1~test*5 f2~test*3x2`), a held-out big-field test;
  - both fields whole (24–30 acres), labelled "partly seen in training", because 80% of their area was trained on.

**Why tiled real data rather than synthetic big fields.** The whole real fields can't be trained on, because they contain the test regions. The largest real training field is therefore 21.5 acres. Laying a training region out 2–3 times gives 25–40-acre fields whose every plant comes from the drone maps, with the right block density (40–45 blocks per acre, like the real fields) and the same trips, turns and battery swaps a big field needs. The older synthetic square fields have smooth made-up texture instead.

## The data (2024, DJI Mavic 3M at 33 m)

| | Field 1 ("Onions") | Field 2 ("Ashley") |
|---|---|---|
| Size (GPS outline) | 30.0 acres | 23.7 acres |
| Beds (found in the images) | 307 beds, ~1.92 m apart, running NW–SE | 104 beds, ~1.82 m apart, running ENE |
| Flights | Jan 31, Feb 14, Mar 7, Mar 18, Apr 4, Apr 17 | same |
| Ground truth | 50 sample points (count, weight, yield, bulb size) | 50 sample points |
| Role | 80% training, 20% test region | 80% training, 20% test region |

## Drone → robot

- **Where the robot measures.** It drives along the beds, so the maps are sampled per real bed, every 0.508 m (6 plant spots per 10-ft block).
- **Finding the beds.** Beds are straight along the field but unevenly spaced across it (1.6–2.2 m between neighbours, from how the bed shaper laid them). So every furrow is found in the images, and a bed is the strip between two furrows. Each flight is registered to the April 4 flight first, because the dates' maps are offset by a few decimetres.
- **What NDVI means.** The robot sees leaves, not the sandy soil, so NDVI is taken from plant pixels only. Only relative differences carry over, because the robot's NDVI scale differs from the drone's (`DRONE_NDVI_GAIN`, which needs a side-by-side measurement).
- **Height.** The Metashape DEMs are 0.4–0.7 m per pixel. That gives 5–12 cm of apparent "canopy height" where real onions are 20–55 cm, so it's mostly bed relief and can't be used. Instead, height = the typical height for the growth stage (`STAGE_HEIGHT_CM`) × (canopy cover / median)^0.5.
- **Plant size and NDVI both matter.** At the lab's sample points, canopy size predicts yield most consistently (r 0.2–0.6 from February on). Plant NDVI predicts well early in the season in Field 2 (r ≈ 0.6).
- **Individual plants.** From 33 m, individual onions can't be separated, so a spot holds the local mean. The simulator adds each plant's own deviation on top.
- **January.** The plants are tiny, so leaf NDVI hardly varies; those maps mostly carry plant-size patterns.

## Cameras: how a measurement can go wrong

Height comes from the **Intel RealSense D455** depth camera. NDVI comes from a red + NIR camera: until the lab picks one, the stand-in is the **JAI AD-130GE**, a prism camera whose red and NIR images come through one lens, as used on the BoniRob field robot. Both look straight down from `CAMERA_HEIGHT_M`. The model is built from datasheets and published measurements; `docs/references.md` lists the sources.

| What limits a measurement | How it is modelled | Source |
|---|---|---|
| **Too close** | No depth closer than the minimum distance: 0.52 m at 1280×720, 0.35 m at 848×480, 0.26 m at 640×360 | D455 datasheet |
| **Depth noise** | Grows with distance²: z² × disparity error ÷ (focal length × 95 mm) | Intel white paper; Keselman 2017 |
| **Systematic depth error** | 4.25 mm SD, larger closer than 0.5 m | Servi et al. 2021 |
| **Fill rate in a field** | About 90% × the lab's own quality-vs-distance log × how much of the plant's top the neighbours leave visible | Fan et al.; lab log |
| **Thin leaves** | Leaf tips narrower than about 3 depth pixels give no depth, so plants look shorter. Seedlings far away give almost none. | 7×7 matching window (Keselman) |
| **NIR detail** | Pixel size on the plant, focus blur (lens, aperture, focus distance) and leaf-motion blur (exposure × wind) | JAI manual, thin-lens optics |
| **Leaf edges** | Blurred edge pixels mix leaf with soil and pull NDVI down; this matters for thin (4–20 mm) onion leaves | onion morphology |
| **Glare** | The scan light's specular glare near one spot reads NDVI low; moving off it helps | Krafft et al. 2024 |
| **Leaf angle and view** | An error fixed for one camera position: shooting again from the same spot doesn't average it out, another spot does | Zhang 2022; Huang 2018 |
| **Calibration** | One NDVI offset per mission | Stamford et al. 2023 |

The robot's live quality readouts are now numbers a real robot can compute: the **depth fill rate** on the plant, and the **share of clean leaf pixels** in the NIR image.

**What each camera sees.** The cameras are mounted with their wide side across the bed. Each camera's footprint on the plant tops shrinks as the plants grow. A plant that is off-centre along the bed can fall outside a view. Plants also sit up to 15 cm to the side of the camera line. The narrow NIR view is usually the first to miss.

### Positioning: the robot picks where to take the picture

1. **Arriving.** The robot stops about 20 cm before the plant (± jitter). The plant's true spot is only known to ±8 cm from the GPS map.
2. **Previews.** While driving in, it has seen live previews from the positions it passed (−30…−10 cm). A preview gives the depth fill rate and the clean-leaf share. Previews are noisier while moving.
3. **Shooting.** The policy then chooses SHOOT at −30…+30 cm or DONE. Each 10-cm move costs time and energy. A careful shot averages frames for `CAREFUL_AVERAGE_S`.
4. **When a plant counts.** A plant counts as measured only once it has an NDVI reading. Height is added when the depth camera resolves the plant.
5. **Fallbacks.** DONE is allowed only after a reading. If two shots give no NDVI, the robot centres on the YOLO box. After 6 shots without a reading it gives up and takes the next plant.
6. **Masks.** The first shot from where the robot stands is always allowed. Moving or shooting again is allowed only within the spare energy, so positioning can never cost a 10-ft sample.
7. **The rules** (baselines and warm start) position this way:
   - quick plants are shot where the robot stops;
   - careful plants are shot from the best depth view seen;
   - if the NIR image is poor, the rules add a second shot from the best NIR view seen.

The energy plan costs every plant with the robot's own running averages of moves and shots, so it always leaves enough for the minimum protocol.

```
python -m amiga_scout cameras                       # facts, limits, per-shot errors by growth stage + figures/camera_curves.png
python -m amiga_scout cameras --set camera_height_m=0.9 --set depth_resolution=1280x720   # try another setup
python -m amiga_scout mount                         # which fixed camera height, depth resolution and NIR lens to use
```

### Camera height (`mount`)

`mount` uses the real plant heights of all 12 maps: the tallest 1% are ≥ 65 cm, or 70 cm on Apr 17. For each height and resolution it checks three things:

- that no plant is closer than the D455 minimum;
- that the depth, RGB and NIR views cover the 1.4-m canopy across the bed;
- the per-shot errors at every growth stage. These are RMS errors, so a steady offset counts as well as the scatter.

| Setup | Too close | Rows covered | Careful plant: height RMS | NDVI RMS (worst stage) | No height |
|---|---|---|---|---|---|
| **1.50 m, D455 1280×720, NIR 2.8 mm f/2 focused at 1.31 m** (the default in `config.py`) | 0% | yes (depth from 1.40 m, NIR from 1.50 m) | 1.2 cm | 0.036 (0.063, Jan) | 0.3% |
| 1.85 m, D455 1280×720, NIR 4 mm f/2 focused at 1.66 m (`mount`'s best score) | 0% | yes | 1.2 cm | 0.030 (0.046, Jan) | 1.6% |
| 1.10 m, D455 1280×720 | 6.8% of plants | no | — | — | — |
| 1.10 m, D455 848×480, NIR 6 mm f/4 focused at 0.7 m (the old setup) | 0% | no: tall plants fall outside the narrow views | — | — | — |

**Why 1.5 m is the default.** It is the lowest mount at which every camera sees the whole canopy across the bed. A lower mast is sturdier, and the scan light is closer to the plants.

**What limits NDVI.** Covering 1.4 m with the NIR camera's 1,296 pixels means pixels of about 1 mm or more on the plants. Thin January leaves (about 5 mm) then mix with soil at their edges, so NDVI reads about 0.05 low. A higher mount with a longer lens makes the seedlings' pixels a little finer, so the seedling error drops: `mount` scores 1.85 m best. The model does not include the costs of a taller mount: less light from the lamp, mast sway, and the robot's height.

To go back to the old 1.1-m setup: `--set camera_height_m=1.1 --set depth_resolution=848x480 --set nir_lens_focal_mm=6 --set nir_lens_fnumber=4 --set nir_focus_m=0.7`.

## How the policy is trained

**Why not PPO alone.** Measured on this simulator:

- One block decision changes a mission's score by only 0.001–0.02 points. The measurement noise that PPO sees when it compares two decisions is 0.1–0.5 points, so it needs a huge number of missions to learn which choice is better.
- PPO's value estimate could not predict the returns within a mission (it explained less than predicting the average would), so its learning signal was mostly noise.
- Trained that way, the RL ended up between the rules everywhere and was best in no condition.

**What `train` does instead (simulation-based policy iteration with rollouts):**

1. **Run training missions** with the current policy, and at randomly picked decisions (10% of blocks) try **every legal choice** in a copy of the mission (`fork`).
2. **Use the same measurement noise in every copy.** Each plant's n-th shot gets the same random numbers whichever copy takes it (common random numbers, as in PEGASUS and TRPO's "vine"). This cut the noise of a comparison 25–650×.
3. **Value each choice** by the accuracy it adds and the hours it costs over the next 10 blocks, with the base rule doing those blocks. The protocol plan for the rest of the route is added, so a battery swap a choice makes necessary counts too. The bug zapper counts as its average share of the work.
4. **An ensemble of 5 networks learns these values from what the robot knows** (its observation). The accuracy a choice adds is learned **per block's share of the field** (× blocks / 1000) and scaled back by the field's own number of blocks when deciding. On a bigger field each block is a smaller share of the map, so the same choice adds proportionally less accuracy; this way the networks don't have to learn that from field size, or guess it for sizes they haven't seen. Each network is fitted on a different resample of the training missions, and their average decides. Accuracy and hours are kept apart, and the policy takes the choice with the best accuracy − `HOUR_VALUE_POINTS` × hours, so the same model can be run for any value of an hour. Every mission contributes the same number of valued decisions (`--per-mission 40`), so small and big fields count equally.
5. **Repeat for 3 rounds.** Each round adds the moments the newer policy visits and keeps earlier data. `curve` compares the rounds on whole missions and picks the best.

This is one improvement step over the base rule, repeated on the states the new policy reaches (Bertsekas 2020; Lagoudakis & Parr 2003). Improving on a policy by simulating from it guarantees at least that policy's score, up to the network's error.

- **Check:** choosing directly by these simulations (which sees the true field, so it's only a check) beat every rule by a wide margin on 5-acre training pieces (spots 89.4 vs 87.7 for every-plant-quick; patches 87.1 vs 84.1).
- **Early result:** one round on 9,000 valued decisions beat every single rule overall on new selection missions (76.5 vs 75.9 for the best rule).

**What went wrong in the first full run, and the fix.** The first model (one network, examples in proportion to field size) scored 0.9 points below "every plant, quick" on the 5-acre test regions: it measured 5.1 plants per block and chose careful in 17% of blocks. The checks showed:

- **The valued choices were right.** At the same decision moments, the training labels valued 6 plants vs 1 at +0.055 points, and the policy's own future over 30 blocks gave +0.056.
- **The network was the problem.** It learned the big average trade-offs, but shrank the small differences between neighbouring choices (the 5th plant: +0.010 instead of +0.027; careful vs quick: −0.007 instead of −0.025). So choosing between near-equal options was decided by its random errors. It picked the truly best option only 17% of the time, no better than chance.
- **Few examples from small fields.** Only 10% of the examples came from 5-acre fields, because a big field has more blocks.

The fix, tested on the same 12,759 valued decisions against selection missions:

| Version | 5-acre pieces vs every plant, quick | Whole regions vs minimum |
|---|---|---|
| One network (as before, balanced data) | −1.20 ± 0.74 | +2.16 ± 0.67 |
| **5-network ensemble (now the default)** | **−0.06 ± 0.07** (a tie) | **+2.43 ± 0.78** |
| + values smooth in the number of plants | −1.27 ± 0.80 | +1.70 ± 1.04 |
| + cautious choice (mean − disagreement) | −4.47 ± 1.70 | +1.22 ± 0.49 |

With the ensemble, the policy matches the best rule on small fields and beats both rules on big ones. The last two variants were worse, so they are off by default (`--value-shape`, `--caution`).

**Big fields, and the final design (tested on the same data, fresh missions):**

- **Big fields made of real drone data** (tiled training regions, 25–40 acres) were at least as good as synthetic square fields in every comparison, and they are real texture.
- **Two network sets are fitted on the same valued decisions:**
  - *raw*: accuracy learned in points. Best on small fields, where values are large.
  - *per-block*: accuracy learned per block's share of the field. Best on big fields and beyond the sizes seen. On its own it cut careful choices on 24–30-acre fields from 10–15% to 1%.
- **The robot blends their values by field size:** all raw up to 300 blocks (~7 acres), all per-block from 900 blocks (~21 acres), linear in between (`--blend-blocks`).
- **It takes careful only when the networks agree:** a careful choice must beat the best quick choice network by network, by more than its spread (`--careful-margin 1`). This removed careful choices the data didn't support.

| Selection missions, vs the best rule for that size | 5-acre pieces | 10–15-acre pieces | Whole fields (24–30 acres) |
|---|---|---|---|
| Raw networks + careful rule | −0.16 ± 0.16 | +0.27 | +1.46 ± 1.03 |
| Per-block networks + careful rule | −0.86 ± 0.41 | +0.27 | +1.78 ± 0.96 |
| **Blend by field size (the default)** | **−0.16 ± 0.16** | **+0.29** | **+1.78 ± 0.96** |

**The value of an hour.** The model learns accuracy and hours separately, so the same model runs at any `hour_value_points`. Tested at 0.5–3 points per hour:

- **Small fields:** measuring every plant stays best at every value tested, and the RL ties it (−0.2 to −0.6).
- **Big fields, 1 point per hour or less:** the RL beats every rule (+1.4 at 0.5, +2.6 at 1).
- **Big fields, 2–3 points per hour:** measuring the minimum is almost optimal, and the RL becomes it (−0.6 and −0.2, within noise).

A higher time value doesn't make the policy better; it makes "measure the minimum" the answer on more fields, leaving less to learn. The time value should come from the lab: how many map-accuracy points one robot hour is worth.

**Positioning** is learned the same way, one level down: every place to shoot from is tried at sampled plants. The learned positioning is kept only if it beats the hand-written one on held-out missions. At 1.5 m it doesn't, because the best spot depends on each plant's hidden leaf angles, so the hand-written positioning stays.

**Accuracy is scored block by block.** Each 10-ft block, each part and each plant is scored on its own and averaged, and the counts of finds and false alarms are not cut off at 0. Before, the block and part terms used one error of the whole field with a floor at 0, and the stress term was floored at 0. So early measurements often changed the score by nothing: on spots fields, finding stressed plants early was worth 0 while false alarms outnumbered the finds. The weights are unchanged (blocks 25%, stress 30%, parts 25%, protocol 20%).

## Field types ("mixed" fields)

The real fields look healthy, so problems are added on top of the real maps, with exact ground truth:

- `as_is`: the real map only.
- `patches`: pest or disease foci added.
- `spots`: isolated stressed plants added.
- `poorer`: the whole field is poorer than the robot assumes.

## Open issues and values still to measure (PLACEHOLDER in `config.py`)

- Power draw, and the time for each nudge and shot.
- The robot-vs-drone NDVI scale: measure at the 50 sample points of each field.
- Onion height at each growth stage, measured with the D455.
- The final NIR camera and its real noise and glare; the Amiga's real speeds, move and shot times, and swap time.
- **Spots fields:** the robot's map overreacts when a single isolated stressed plant is measured, dragging its neighbours into false alarms. A robust (Student-t) plant model is the principled fix.
- **Next step:** measure the placeholders on the real Amiga, retrain, then run a shadow-mode field trial (the robot follows the protocol rule while the RL's choices are logged and compared).

## Acknowledgements and references

The design builds on HAM-PPO (Khosravi et al. 2025), Gaussian-process active sampling for phenotyping (Kumar et al. 2019), rollout and policy iteration (Bertsekas 2020; Lagoudakis & Parr 2003), and common random numbers in policy search (Ng & Jordan 2000; Schulman et al. 2015). The camera model is built from the Intel D455 and JAI AD-130GE datasheets and published measurements. `docs/references.md` lists every source and what it supports.

## Contact

Saklain Niam: saqlineniam@gmail.com (data access, questions, collaboration).
