"""A policy learned by simulation-based policy iteration (amiga_scout.rollout).

Two small networks, one per decision level, predict from what the robot knows (its observation) how much every
choice adds to ACCURACY and how many hours it costs, compared with the protocol choice (QUICK_1 at a block,
SHOOT_HERE - or DONE once the plant has its reading - at a plant). The robot takes the choice with the best
accuracy - HOUR_VALUE_POINTS x hours. Accuracy and time are kept apart, so the same model can be run for any
value of an hour (`--set hour_value_points=...`).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..actions import ACTIONS, decode
from .rules import BASELINES, Policy, position_rule

FORMAT = "amiga_scout rollout policy v1"


class ChoiceModel:
    """One decision level: observation -> (accuracy gain, hours) of each of its actions, relative to the
    reference action. An ensemble: the prediction is the average of several networks fitted on different resamples
    of the training missions - with noisy targets, the best of 12 guesses from ONE network picks its random errors
    (the optimizer's curse); the average of several is steadier. numpy forward pass: fast enough to be called
    inside millions of simulated decisions."""

    def __init__(self, actions, ref, mean, std, acc_scale, hrs_scale, nets, info=None, caution=0.0, design=None,
                 per_block=False):
        self.actions, self.ref = np.asarray(actions, int), int(ref)
        self.mean, self.std = np.asarray(mean, np.float64), np.asarray(std, np.float64)
        self.acc_scale, self.hrs_scale = float(acc_scale), float(hrs_scale)
        if nets and isinstance(nets[0], tuple):          # one network (older files)
            nets = [nets]
        self.nets = [[(np.asarray(W, np.float64), np.asarray(b, np.float64)) for W, b in net] for net in nets]
        self.info = dict(info or {})
        self.caution = float(caution)     # choose by mean - caution x disagreement of the networks (0 = by the mean)
        # design: the networks output a few parameters per situation and value(action) = parameters @ design[action]
        # (block level: smooth in the number of plants - see amiga_scout.rollout.block_design); None = one output
        # per action
        self.design = None if design is None else np.asarray(design, np.float64)
        # per_block: the networks predict the accuracy gain in units of one block's share of the field (x blocks /
        # 1000), which is about the same on a 5-acre and a 40-acre field; it is scaled back by the field's own
        # number of blocks when deciding - so field size does not have to be learned (or guessed beyond the sizes
        # trained on)
        self.per_block = bool(per_block)

    def _forward(self, h, net):
        for W, b in net[:-1]:
            h = np.tanh(h @ W + b)
        W, b = net[-1]
        return h @ W + b

    def members(self, obs, blocks=None):
        """(accuracy gain, hours) of every action from each network: arrays (networks, ..., actions). blocks: the
        field's number of 10-ft blocks (needed when per_block)."""
        h = (np.asarray(obs, np.float64) - self.mean) / self.std
        out = np.array([self._forward(h, net) for net in self.nets])
        if self.design is not None:
            q = self.design.shape[1]
            acc, hrs = (out[..., :q] @ self.design.T) * self.acc_scale, (out[..., q:] @ self.design.T) * self.hrs_scale
        else:
            n = len(self.actions)
            acc, hrs = out[..., :n] * self.acc_scale, out[..., n:] * self.hrs_scale
        if self.per_block:
            if blocks is None:
                raise ValueError("this model predicts per block: pass the field's number of blocks")
            acc = acc / (np.asarray(blocks, np.float64)[..., None] / 1000.0)
        return acc, hrs

    def predict(self, obs, blocks=None):
        """(accuracy gain, hours) of every action of this level for one observation (or a batch): the ensemble mean."""
        acc, hrs = self.members(obs, blocks)
        return acc.mean(0), hrs.mean(0)

    def scores(self, obs, hour_value, blocks=None):
        """What the choice is made on: mean over the networks of accuracy - hour_value x hours, minus `caution` x
        their disagreement (a lower confidence bound: a choice the data does not clearly support does not win by luck)."""
        acc, hrs = self.members(obs, blocks)
        v = acc - hour_value * hrs
        return v.mean(0) - (self.caution * v.std(0) if self.caution and len(self.nets) > 1 else 0.0)

    def best(self, env, hour_value):
        """The legal action with the best score (see scores; the reference if none is legal)."""
        m = env.action_masks()
        score = self.scores(env._obs(), hour_value, env.NBv)
        legal = m[self.actions]
        if not legal.any():
            return self.ref if m[self.ref] else int(np.flatnonzero(m)[0])
        return int(self.actions[np.flatnonzero(legal)[np.argmax(score[legal])]])

    def to_dict(self):
        return dict(actions=self.actions.tolist(), ref=self.ref, mean=self.mean, std=self.std,
                    acc_scale=self.acc_scale, hrs_scale=self.hrs_scale, nets=self.nets, info=self.info,
                    caution=self.caution, design=self.design, per_block=self.per_block)

    @classmethod
    def from_dict(cls, d):
        return cls(d["actions"], d["ref"], d["mean"], d["std"], d["acc_scale"], d["hrs_scale"],
                   d.get("nets") or d.get("weights"), d.get("info"), d.get("caution", 0.0), d.get("design"),
                   d.get("per_block", False))


class BlockChooser:
    """How the robot picks a block choice from two sets of networks fitted on the same valued decisions:
        raw        accuracy learned in points - best on small fields, where the values are large
        per-block  accuracy learned per block's share of the field - best on big fields (and beyond the sizes seen)
    Their values are blended by the field's number of blocks: all raw up to blend[0], all per-block from blend[1],
    linear in between. The robot takes the best quick choice, and switches to the best careful choice only when the
    networks agree it pays: its advantage, network by network, has mean - careful_margin x spread > 0 (careful choices
    the data does not clearly support stopped winning by the networks' random errors)."""

    def __init__(self, raw=None, per_block=None, blend=(300, 900), careful_margin=1.0, info=None):
        self.raw, self.pb = raw, per_block
        self.blend, self.careful_margin = tuple(blend), float(careful_margin)
        self.info = dict(info or {})
        m = raw or per_block
        self.actions, self.ref = m.actions, m.ref

    def weight(self, blocks):
        """Share of the per-block networks for a field of this many blocks."""
        lo, hi = self.blend
        if self.raw is None:
            return 1.0
        if self.pb is None:
            return 0.0
        return float(np.clip((blocks - lo) / max(hi - lo, 1e-9), 0.0, 1.0))

    def members_value(self, obs, hour_value, blocks):
        """Value (accuracy - hour_value x hours) of every action, per network: (networks, ..., actions)."""
        w = self.weight(blocks) if np.ndim(blocks) == 0 else np.clip(
            (np.asarray(blocks, np.float64) - self.blend[0]) / max(self.blend[1] - self.blend[0], 1e-9), 0, 1)[..., None]
        out = 0.0
        for model, share in ((self.raw, 1.0 - w), (self.pb, w)):
            if model is None or (np.ndim(share) == 0 and share == 0.0):
                continue
            acc, hrs = model.members(obs, blocks)
            out = out + share * (acc - hour_value * hrs)
        return out

    def predict(self, obs, blocks=None):
        """(accuracy gain, hours) blended by field size (the ensemble mean) - for reports."""
        w = np.clip((np.asarray(blocks, np.float64) - self.blend[0]) / max(self.blend[1] - self.blend[0], 1e-9), 0, 1)[..., None]
        if self.raw is None or self.pb is None:
            return (self.raw or self.pb).predict(obs, blocks)
        ar, hr = self.raw.predict(obs, blocks)
        ap, hp = self.pb.predict(obs, blocks)
        return (1 - w) * ar + w * ap, (1 - w) * hr + w * hp

    def scores(self, obs, hour_value, blocks=None):
        return np.mean(self.members_value(obs, hour_value, blocks), axis=0)

    def best(self, env, hour_value):
        m = env.action_masks()
        v = self.members_value(env._obs(), hour_value, env.NBv)       # (networks, actions)
        mean = v.mean(0)
        legal = m[self.actions]
        if not legal.any():
            return self.ref if m[self.ref] else int(np.flatnonzero(m)[0])
        careful = np.array([decode(a)[0] == "careful" for a in self.actions])
        quick = np.flatnonzero(legal & ~careful)
        care = np.flatnonzero(legal & careful)
        iq = int(quick[np.argmax(mean[quick])]) if len(quick) else int(care[np.argmax(mean[care])])
        if len(care) and len(quick):
            ic = int(care[np.argmax(mean[care])])
            adv = v[:, ic] - v[:, iq]
            if adv.mean() - self.careful_margin * adv.std() > 0:
                return int(self.actions[ic])
        return int(self.actions[iq])

    def to_dict(self):
        return dict(kind="chooser", raw=self.raw.to_dict() if self.raw else None,
                    per_block=self.pb.to_dict() if self.pb else None, blend=list(self.blend),
                    careful_margin=self.careful_margin, info=self.info)

    @classmethod
    def from_dict(cls, d):
        return cls(ChoiceModel.from_dict(d["raw"]) if d.get("raw") else None,
                   ChoiceModel.from_dict(d["per_block"]) if d.get("per_block") else None,
                   d.get("blend", (300, 900)), d.get("careful_margin", 1.0), d.get("info"))


def block_model_from_dict(d):
    return BlockChooser.from_dict(d) if d.get("kind") == "chooser" else ChoiceModel.from_dict(d)


def save_policy(path, block=None, plant=None, base="minimum", info=None):
    """Write a learned policy (block and/or plant model) to path (.pt)."""
    import torch as th
    from ..env.observation import OBS_NAMES
    th.save(dict(format=FORMAT, obs_names=list(OBS_NAMES), actions=list(ACTIONS), base=base,
                 block=block.to_dict() if block else None, plant=plant.to_dict() if plant else None,
                 info=dict(info or {})), str(path))


def load_policy_dict(path):
    import torch as th
    d = th.load(str(path), weights_only=False)
    if not isinstance(d, dict) or d.get("format") != FORMAT:
        raise SystemExit(f"{path} is not a rollout policy file")
    return d


class RolloutPolicy(Policy):
    """The learned policy: block choices from the block model (the base rule if there is none yet) and positioning
    from the plant model (the hand-written positioning if there is none yet)."""

    def __init__(self, path):
        from ..env.observation import OBS_NAMES
        path = Path(path)
        if path.suffix != ".pt":
            path = path.with_name(path.name + ".pt")
        d = load_policy_dict(path)
        self.name = f"RL: {path.stem}"
        if d["obs_names"] != list(OBS_NAMES) or d["actions"] != list(ACTIONS):
            raise SystemExit(f"{self.name} was trained on an older version of the simulator - retrain it with `train`")
        self.block = block_model_from_dict(d["block"]) if d.get("block") else None
        self.plant = ChoiceModel.from_dict(d["plant"]) if d.get("plant") else None
        self.base = BASELINES[d.get("base") or "minimum"]()
        self.info = d.get("info", {})
        self.positioner = self._position if self.plant else position_rule

    def block_act(self, env):
        if self.block is None:
            return self.base.block_act(env)
        return self.block.best(env, env.cfg.hour_value_points)

    def plant_act(self, env):
        return self.positioner(env)

    def _position(self, env):
        return self.plant.best(env, env.cfg.hour_value_points)
