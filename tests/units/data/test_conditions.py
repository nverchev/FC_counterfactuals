"""Tests for the conditioning probabilities."""

import math

import torch

from src.config.specs import ExperimentConfig
from src.data.split import DataSplit
from src.data.conditions import condition_split, rescale


class TestRescale:
    """Test the rescaling of the probabilities towards the middle."""

    def test_fixed_points(self) -> None:
        """0, 0.5 and 1 stay where they are."""
        probabilities = torch.tensor([0.0, 0.5, 1.0])

        result = rescale(probabilities, 3)

        torch.testing.assert_close(result, probabilities)

    def test_one_step(self) -> None:
        """A step applies arccos(1 - 2p) / pi."""
        result = rescale(torch.tensor([0.1]), 1)

        torch.testing.assert_close(result, torch.tensor([math.acos(0.8) / math.pi]))

    def test_steps_spread_saturated_probabilities(self) -> None:
        """Each step moves a probability near 0 towards the middle, keeping the order."""
        probabilities = torch.tensor([1e-4, 1e-2, 0.3])

        once, twice = rescale(probabilities, 1), rescale(probabilities, 2)

        assert torch.all(twice > once) and torch.all(once > probabilities)
        assert torch.all(twice.diff() > 0)


def test_condition_split(example_config: ExperimentConfig) -> None:
    """The split receives the probabilities of the classifier, rescaled as configured."""
    split = DataSplit(torch.zeros(3, 2), torch.zeros(3, 1))
    cfg_classification = example_config.classification
    steps = cfg_classification.rescale_steps[cfg_classification.model.name]

    conditioned = condition_split(lambda features: torch.full((len(features), 1), 0.2), split)

    torch.testing.assert_close(conditioned.conditions, rescale(torch.full((3, 1), 0.2), steps))
