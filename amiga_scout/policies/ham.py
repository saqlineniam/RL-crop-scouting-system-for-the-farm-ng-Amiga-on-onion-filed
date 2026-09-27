"""
amiga_scout.policies.ham - Hierarchical Action Masking policy for MaskablePPO (HAM-PPO style)
==================================================================================

Implements the conditional action tree from Khosravi et al. (2025) on top of
sb3-contrib's MaskablePPO, without changing the environment's Discrete action space:

    level 0 (b0):  which BRANCH       e.g. MOVE  or  SENSE
    level 1 (b1):  which LEAF inside  e.g. NEXT_PLANT / QUICK_SCAN / CAREFUL_SCAN ...

    pi(a | s) = P(b0 | h0, s) * P(b1 | b0, h1, s)                    (paper Eq. 16)

The actor has two heads on a shared body: a branch head (one logit per branch) and a
leaf head (one logit per leaf). Masks are applied at BOTH levels:
    h0: a branch is legal only if at least one of its leaves is legal
    h1: inside the chosen branch, only legal leaves are renormalised
This is different from simply masking a flat softmax: here the branch probability is
decided first, then the leaf probability is renormalised only within that branch.

The combined log-probability is handed to PPO as one categorical over the flat action
list, so save/load, predict(action_masks=...), and ActionMasker all work unchanged.
The chain rule makes the entropy of that categorical equal H(b0) + E[H(b1|b0)].
"""
from __future__ import annotations

import torch as th
from torch import nn

from sb3_contrib.common.maskable.distributions import MaskableCategorical, MaskableCategoricalDistribution
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

NEG = -1e8


class _TwoHeads(nn.Module):
    """Branch head + leaf head on the shared actor latent; outputs are concatenated."""

    def __init__(self, latent_dim, n_branches, n_leaves):
        super().__init__()
        self.branch_head = nn.Linear(latent_dim, n_branches)
        self.leaf_head = nn.Linear(latent_dim, n_leaves)

    def forward(self, latent):
        return th.cat([self.branch_head(latent), self.leaf_head(latent)], dim=-1)


class HierarchicalCategoricalDistribution(MaskableCategoricalDistribution):
    def __init__(self, action_groups):
        super().__init__(len(action_groups))
        self.groups = th.as_tensor(list(action_groups), dtype=th.long)
        self.n_groups = int(self.groups.max()) + 1
        member = th.zeros(self.action_dim, self.n_groups, dtype=th.bool)
        member[th.arange(self.action_dim), self.groups] = True
        self.member = member
        self._branch = self._leaf = None

    def proba_distribution_net(self, latent_dim):
        return _TwoHeads(latent_dim, self.n_groups, self.action_dim)

    def proba_distribution(self, action_logits):
        out = action_logits.view(-1, self.n_groups + self.action_dim)
        self._branch, self._leaf = out[:, :self.n_groups], out[:, self.n_groups:]
        self.distribution = MaskableCategorical(logits=self._compose(None))
        return self

    def apply_masking(self, masks):
        if masks is None:
            self.distribution = MaskableCategorical(logits=self._compose(None))
            return
        m = th.as_tensor(masks, dtype=th.bool, device=self._leaf.device).reshape(-1, self.action_dim)
        self.distribution = MaskableCategorical(logits=self._compose(m), masks=m)

    def _compose(self, m):
        dev = self._leaf.device
        groups, member = self.groups.to(dev), self.member.to(dev)
        leaf, branch = self._leaf, self._branch
        if m is not None:
            leaf = th.where(m, leaf, th.full_like(leaf, NEG))
        # log P(b1 | b0): softmax of the leaf logits within each branch (h1 applied above)
        x = th.where(member.unsqueeze(0), leaf.unsqueeze(-1), th.full((1, *member.shape), NEG, device=dev))
        lse = th.logsumexp(x, dim=1)                              # (B, n_groups)
        leaf_lp = leaf - lse[:, groups]
        # log P(b0): a branch is legal if any of its leaves is legal (h0)
        if m is not None:
            legal_branch = (m.unsqueeze(-1) & member.unsqueeze(0)).any(dim=1)
            branch = th.where(legal_branch, branch, th.full_like(branch, NEG))
        branch_lp = th.log_softmax(branch, dim=-1)
        return branch_lp[:, groups] + leaf_lp

    def branch_probs(self):
        """P(b0) after masking, for inspection/plots."""
        p = self.distribution.probs
        out = th.zeros(p.shape[0], self.n_groups, device=p.device)
        return out.index_add_(1, self.groups.to(p.device), p)


class HierarchicalMaskablePolicy(MaskableActorCriticPolicy):
    """MaskablePPO policy whose actor is a two-level conditional action tree."""

    def __init__(self, observation_space, action_space, lr_schedule, action_groups=None, **kwargs):
        if action_groups is None:
            raise ValueError("HierarchicalMaskablePolicy needs policy_kwargs=dict(action_groups=[...])")
        self.action_groups = list(action_groups)
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def _build(self, lr_schedule):
        self.action_dist = HierarchicalCategoricalDistribution(self.action_groups)
        super()._build(lr_schedule)

    def _get_constructor_parameters(self):
        data = super()._get_constructor_parameters()
        data.update(action_groups=self.action_groups)
        return data
