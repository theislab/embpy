"""Simple non-deep baselines for the perturbation world model.

The baselines deliberately stay non-parametric or low-capacity so they
are fast to fit, easy to debug, and serve as a sanity floor for the
world model. We avoid CPA / chemCPA / scGPT etc. on purpose.

All baselines share the :class:`Baseline` interface and operate in
*gene-expression space*: ``fit(train)`` consumes a
:class:`BaselineTrainData` and ``predict(perts, control_template)``
returns predictions of shape ``(n_perturbations, n_genes)``.
"""

from __future__ import annotations

from .additive import AdditiveBaseline
from .base import Baseline, BaselineTrainData
from .control_mean import ControlMeanBaseline
from .identity import IdentityBaseline
from .linear import LinearRegressionBaseline
from .mean import MeanBaseline

ALL_BASELINES: dict[str, type[Baseline]] = {
    "identity": IdentityBaseline,
    "control_mean": ControlMeanBaseline,
    "mean": MeanBaseline,
    "additive": AdditiveBaseline,
    "linear": LinearRegressionBaseline,
}

__all__ = [
    "ALL_BASELINES",
    "AdditiveBaseline",
    "Baseline",
    "BaselineTrainData",
    "ControlMeanBaseline",
    "IdentityBaseline",
    "LinearRegressionBaseline",
    "MeanBaseline",
]
