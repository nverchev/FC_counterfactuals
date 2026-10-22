"""Defines all enumerated options used in configuration files."""

import enum

__all__ = [
    'Classifiers',
    'ClipCriterion',
    'CounterfactualTargets',
    'Datasets',
    'GradOp',
    'Schedulers',
]


# ---------------------------------------------------------------------------
# Project Specific
# ---------------------------------------------------------------------------


class Classifiers(enum.StrEnum):
    """Classifiers whose probabilities condition the counterfactual VAE.

    ``AE`` is the MLP whose encoder starts from the weights of the pretrained autoencoder.
    """

    SVC = enum.auto()
    MLP = enum.auto()
    AE = enum.auto()


class CounterfactualTargets(enum.StrEnum):
    """Probabilities the counterfactuals aim at.

    ``OPPOSITE`` aims at the class the classifier does not predict (0 or 1), ``HALF`` at the
    decision boundary (0.5).
    """

    OPPOSITE = enum.auto()
    HALF = enum.auto()


class Datasets(enum.StrEnum):
    """Dataset choices."""

    ABIDE = enum.auto()


# ---------------------------------------------------------------------------
# Generic options available in drytorch
# ---------------------------------------------------------------------------


class GradOp(enum.StrEnum):
    """Gradient operation names."""

    PARAM_NORMALIZER = enum.auto()
    Z_SCORE_NORMALIZER = enum.auto()
    NORM_CLIPPER = enum.auto()
    VALUE_CLIPPER = enum.auto()
    HIST_CLIPPER = enum.auto()
    PARAM_HIST_CLIPPER = enum.auto()


class ClipCriterion(enum.StrEnum):
    """Clipping criterion names for histogram-based gradient clippers."""

    Z_STAT = enum.auto()
    EMA = enum.auto()


class Schedulers(enum.StrEnum):
    """Learning-rate scheduler names."""

    CONSTANT = enum.auto()
    COSINE = enum.auto()
    EXPONENTIAL = enum.auto()
    POLYNOMIAL = enum.auto()
    STEP = enum.auto()
