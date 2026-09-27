"""Small numeric helpers."""
from __future__ import annotations

import numpy as np


def _phi(x):
    """Standard normal CDF (tanh approximation, error < 3e-4)."""
    return 0.5 * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))


def _f1(tp, fp, fn):
    return 1.0 if tp + fp + fn == 0 else 2.0 * tp / (2.0 * tp + fp + fn)
