"""The command line: `python -m amiga_scout <command>` (run from the repository folder)."""
from __future__ import annotations

import argparse
import os

from .commands.cameras import cmd_cameras
from .commands.compare import cmd_compare
from .commands.curve import cmd_curve
from .commands.describe import cmd_describe
from .commands.explain import cmd_explain
from .commands.fieldwise import cmd_fieldwise
from .commands.mount import cmd_mount
from .commands.validate import cmd_validate
from .commands.watch import cmd_watch
from .config import BIG_TEST_FIELDS, DEFAULT_SCENARIO, FIELD, SCENARIOS, SELECT_FIELDS, TEST_FIELD, TRAIN_FIELDS
from .drone.analysis import cmd_analyse
from .drone.beds import cmd_geometry
from .drone.extract import cmd_extract, cmd_extract_all
from .drone.io import DATES, FIELDS
from .policies.rules import BASELINES
from .training import cmd_train


from . import __doc__ as PACKAGE_DOC


def main():
    ap = argparse.ArgumentParser(prog="python -m amiga_scout", description=PACKAGE_DOC,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    default_workers = max(1, min(16, (os.cpu_count() or 2) - 1))

    def common(p, episodes=10, field=None):
        p.add_argument("--episodes", type=int, default=episodes)
        p.add_argument("--set", action="append", help="override a setting: --set name=value (repeatable)")
        p.add_argument("--seed", type=int, default=1000)
        p.add_argument("--field", default=field, help=f"field or pool, e.g. f1_05, f2, f2@10 (random 10-acre pieces), 'f1 f1@5' (default {field or FIELD})")

    p = sub.add_parser("describe"); common(p); p.set_defaults(fn=cmd_describe)
    p = sub.add_parser("cameras", help="the camera model: datasheet facts, limits, per-shot errors, curves figure")
    p.add_argument("--set", action="append", help="override a setting: --set name=value (repeatable)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shots", type=int, default=1500, help="simulated shots per growth stage")
    p.add_argument("--out", default=None, help="figure path (default figures/camera_curves.png)")
    p.add_argument("--no-figure", action="store_true")
    p.set_defaults(fn=cmd_cameras)
    p = sub.add_parser("mount", help="which fixed camera height, depth resolution and NIR lens work all season")
    p.add_argument("--set", action="append", help="override a setting: --set name=value (repeatable)")
    p.add_argument("--min", type=float, default=0.9, help="lowest camera height to check (m)")
    p.add_argument("--max", type=float, default=2.0, help="highest camera height to check (m)")
    p.add_argument("--step", type=float, default=0.05)
    p.add_argument("--shots", type=int, default=250, help="simulated plants per growth stage and height")
    p.add_argument("--max-too-close", type=float, default=0.01, help="allowed share of plants inside Min-Z")
    p.add_argument("--all", action="store_true", help="print every height (default: every 0.1 m)")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_mount)
    for name in ("baselines", "compare"):
        p = sub.add_parser(name); common(p)
        p.add_argument("--policies", nargs="*", choices=list(BASELINES))
        p.add_argument("--models", nargs="*", help="trained model paths (without .zip)")
        p.add_argument("--scenarios", nargs="*", default=[DEFAULT_SCENARIO] if name == "baselines" else list(SCENARIOS),
                       choices=list(SCENARIOS))
        p.add_argument("--brief", action="store_true", help="print only the overall-score ranking")
        p.add_argument("--workers", type=int, default=default_workers, help="processes to run missions in parallel")
        p.set_defaults(fn=cmd_compare)
    p = sub.add_parser("validate", help="is a trained model good enough for the field?")
    p.add_argument("--models", nargs="*", help="trained model paths (without .zip)")
    p.add_argument("--test-field", default=TEST_FIELD, help="the held-out test regions (every flight)")
    p.add_argument("--train-field", default=TRAIN_FIELDS, help="the training fields (only to label the other fields)")
    p.add_argument("--other-fields", nargs="*", default=[BIG_TEST_FIELDS, "f1 f2"],
                   help="more fields/pools to check (default: ~30-acre fields made only of the test regions - held out; "
                        "and both fields whole, 80%% of whose area was used in training)")
    p.add_argument("--sizes", type=float, nargs="*", default=[], help="piece sizes (acres) of the test regions to check")
    p.add_argument("--fields", type=int, default=24, help="missions per field type on the test field")
    p.add_argument("--stress-fields", type=int, default=12, help="missions per other field and per stress test")
    p.add_argument("--seed", type=int, default=90000, help="first held-out mission (training never uses these)")
    p.add_argument("--set", action="append", help="override a setting: --set name=value (repeatable)")
    p.add_argument("--workers", type=int, default=default_workers)
    p.set_defaults(fn=cmd_validate)
    p = sub.add_parser("fieldwise", help="the score field by field (every flight date x field type), and what it is made of")
    p.add_argument("--models", nargs="*", help="trained model paths (without .zip)")
    p.add_argument("--rules", nargs="*", default=["minimum", "all_plants_quick"], choices=list(BASELINES),
                   help="hand-written rules to compare with")
    p.add_argument("--field", default=TEST_FIELD, help="field pool, one row per field: f1~test (default: both test "
                                                        "regions), f2, f2~train@10 (pieces) ...")
    p.add_argument("--scenarios", nargs="*", default=list(SCENARIOS), choices=list(SCENARIOS))
    p.add_argument("--missions", type=int, default=4, help="missions per field and field type (different routes and stress)")
    p.add_argument("--seed", type=int, default=90000, help="first mission (training never uses 90000+)")
    p.add_argument("--csv", help="where to save every mission (default models/fieldwise.csv)")
    p.add_argument("--set", action="append", help="override a setting: --set name=value (repeatable)")
    p.add_argument("--workers", type=int, default=default_workers)
    p.set_defaults(fn=cmd_fieldwise)
    p = sub.add_parser("curve", help="score of every training round / checkpoint on selection fields (to pick the best)")
    p.add_argument("--model-path", required=True, help="the model path used in `train` (without .pt / .zip)")
    p.add_argument("--final-steps", type=int, default=0, help="PPO only: steps of the final model (its training --steps)")
    p.add_argument("--every", type=int, default=2, help="PPO only: use every Nth checkpoint (every 250k steps: 2 = every 500k)")
    p.add_argument("--select-fields", nargs="*", default=list(SELECT_FIELDS),
                   help="selection field pools (training regions with new routes and stress - NOT the test regions)")
    p.add_argument("--fields", type=int, default=8, help="selection missions per field pool")
    p.add_argument("--seed", type=int, default=80000, help="first selection mission (validate uses 90000+)")
    p.add_argument("--set", action="append", help="override a setting: --set name=value (repeatable)")
    p.add_argument("--workers", type=int, default=default_workers)
    p.set_defaults(fn=cmd_curve)
    p = sub.add_parser("watch"); common(p, 1)
    p.add_argument("--policy", default="adaptive", help=f"one of {list(BASELINES)} or a model path")
    p.add_argument("--scenario", default=DEFAULT_SCENARIO, choices=list(SCENARIOS))
    p.set_defaults(fn=cmd_watch)
    p = sub.add_parser("explain", help="extract a readable decision tree from a trained model")
    common(p, 5)
    p.add_argument("--policy", required=True, help="model path (or a baseline name, to test the tool)")
    p.add_argument("--depth", type=int, default=4, help="tree depth: 3-5 is readable")
    p.add_argument("--scenario", default=DEFAULT_SCENARIO, choices=list(SCENARIOS))
    p.add_argument("--evaluate", type=int, default=0, help="also run N missions with the tree vs the original")
    p.add_argument("--out", default=None, help="output path without extension")
    p.set_defaults(fn=cmd_explain)
    p = sub.add_parser("train", help="train the RL policy (default: policy iteration with rollouts; or PPO)")
    common(p, field=TRAIN_FIELDS)
    p.add_argument("--method", choices=["rollout", "ppo"], default="rollout",
                   help="rollout: simulation-based policy iteration with common random numbers (default); "
                        "ppo: HAM-PPO (the earlier method, kept for comparison)")
    p.add_argument("--test-field", default=TEST_FIELD, help="refuse to train on this field or region (kept for validate)")
    p.add_argument("--scenario", default="mixed", choices=list(SCENARIOS) + ["mixed"],
                   help="'mixed' = a random field type every episode (default; more robust policy)")
    p.add_argument("--randomize", action="store_true",
                   help="different robot/weather conditions every episode (speed, power, packs, image noise, zapper)")
    p.add_argument("--model-path", default=None,
                   help="where to save (default models/onion_rl for rollout, models/onion_ham for ppo)")
    p.add_argument("--workers", type=int, default=default_workers)
    g = p.add_argument_group("rollout training")
    g.add_argument("--iterations", type=int, default=3, help="rounds of policy improvement for the block decisions")
    g.add_argument("--states", type=int, default=30000, help="block decisions valued per round")
    g.add_argument("--plant-states", type=int, default=20000, help="plant decisions valued to learn positioning (0 = keep the rule)")
    g.add_argument("--horizon", type=int, default=10, help="blocks simulated after a block choice to value it")
    g.add_argument("--salts", type=int, default=2, help="independent draws of the future readings per choice")
    g.add_argument("--base", default="minimum", choices=[n for n in BASELINES if n != "random"],
                   help="the hand-written rule the first round improves on")
    g.add_argument("--continuation", choices=["base", "repeat", "current"], default="base",
                   help="who does the blocks after a valued choice: the base rule; 'repeat' = measured the way the policy "
                        "chose for this block (its own density, the same in every copy); or the policy itself (noisier)")
    g.add_argument("--per-mission", type=int, default=40,
                   help="valued block decisions per mission (the same for small and big fields; 0 = --block-share of blocks)")
    g.add_argument("--members", type=int, default=5, help="networks in the ensemble (their average decides)")
    g.add_argument("--caution", type=float, default=0.0,
                   help="choose by the networks' mean minus this x their disagreement (0 = by the mean; 1 was tested and "
                        "measured too few plants: the networks disagree more about many plants)")
    g.add_argument("--values", choices=["blend", "raw", "per-block"], default="blend",
                   help="block values learned in points (raw: best on small fields), per block's share of the field "
                        "(per-block: best on big fields), or both, blended by field size (default)")
    g.add_argument("--blend-blocks", type=float, nargs=2, default=[300, 900],
                   help="all raw up to the first number of blocks, all per-block from the second (linear between)")
    g.add_argument("--careful-margin", type=float, default=1.0,
                   help="careful only if its advantage, network by network, has mean - this x spread > 0")
    g.add_argument("--value-shape", choices=["free", "smooth"], default="free",
                   help="one value per block choice (default), or smooth in the number of plants (tested: worse)")
    g.add_argument("--block-share", type=float, default=0.1, help="share of block decisions valued in a mission")
    g.add_argument("--plant-share", type=float, default=0.05, help="share of plant decisions valued in a mission")
    g.add_argument("--explore-block", type=float, default=0.2, help="share of random block choices while collecting")
    g.add_argument("--explore-plant", type=float, default=0.3, help="share of random plant choices while collecting")
    g = p.add_argument_group("ppo training")
    g.add_argument("--arch", choices=["ham", "flat"], default="ham")
    g.add_argument("--steps", type=int, default=5_000_000, help="decisions to train on")
    g.add_argument("--ent-coef", type=float, default=0.01, help="exploration bonus (HAM-PPO used 0.02)")
    g.add_argument("--n-envs", type=int, default=16)
    g.add_argument("--checkpoint-steps", type=int, default=250_000)
    g.add_argument("--lr-end", type=float, default=0.1, help="final learning rate as a share of the first (1 = constant)")
    g.add_argument("--warm-start", choices=[n for n in BASELINES if n != "random"], default=None,
                   help="first copy this hand-written rule (e.g. adaptive), then let RL improve on it")
    g.add_argument("--demo-episodes", type=int, default=48, help="missions of the rule to copy from")
    p.set_defaults(fn=cmd_train)
    # ---- the drone survey -> robot-view maps
    p = sub.add_parser("geometry", help="find the beds of each field: exact direction and every furrow")
    p.add_argument("--field", nargs="*"); p.set_defaults(fn=cmd_geometry)
    p = sub.add_parser("extract", help="one flight -> its robot-view map")
    p.add_argument("--field", required=True, choices=list(FIELDS)); p.add_argument("--date", required=True, choices=list(DATES))
    p.set_defaults(fn=cmd_extract)
    p = sub.add_parser("extract-all", help="every flight of both fields (skips maps already made)")
    p.add_argument("--redo", action="store_true"); p.set_defaults(fn=cmd_extract_all)
    p = sub.add_parser("analyse", help="map statistics, yield check at the sample points, figures")
    p.add_argument("--dem", action="store_true", help="also run the DEM height check (~1 min)"); p.set_defaults(fn=cmd_analyse)
    args = ap.parse_args()
    args.fn(args)
