"""Probabilities of the classifier that condition the counterfactual VAE."""

import math
from collections.abc import Callable

import torch

from src.config.experiment import Experiment
from src.data.split import DataSplit

__all__ = [
    'condition_split',
    'rescale',
]


def condition_split(probability: Callable[[torch.Tensor], torch.Tensor], split: DataSplit, /) -> DataSplit:
    """Set the rescaled probabilities of the classifier as conditions of the split.

    Args:
        probability: the trained classifier, from connections to probabilities.
        split: the split to condition.

    Returns:
        The split with its conditions.
    """
    cfg_classification = Experiment.get_config().classification
    steps = cfg_classification.rescale_steps[cfg_classification.model.name]
    return split.with_conditions(rescale(probability(split.features), steps))


def rescale(probabilities: torch.Tensor, steps: int) -> torch.Tensor:
    """Spread probabilities close to 0 or 1 towards the middle.

    Each step applies ``arccos(1 - 2p) / pi``, which keeps 0, 0.5 and 1 and the order of the
    probabilities, so that a saturated classifier still gives the VAE a usable condition.

    Args:
        probabilities: probabilities in [0, 1].
        steps: number of times the map is applied.

    Returns:
        The rescaled probabilities, with the same type as the input.
    """
    rescaled = probabilities.double()
    for _ in range(steps):
        rescaled = torch.arccos(1 - 2 * rescaled) / math.pi

    return rescaled.to(probabilities.dtype)
