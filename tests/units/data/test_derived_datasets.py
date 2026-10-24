"""Tests for the splits carrying derived data."""

import math

import pytest
import torch

from src.config.options import CounterfactualTargets
from src.config.specs import ExperimentConfig
from src.data.derived_datasets import (
    ConditionedSplit,
    CounterfactualSplit,
    ReconstructedSplit,
    get_targets,
    rescale,
)
from src.data.split import DataSplit
from src.module import CounterfactualVAE
from src.data.abide import N_FEATURES
from src.data.structures import Inputs


class _Classifier:
    probabilities: torch.Tensor

    def __init__(self, probabilities: torch.Tensor) -> None:
        self.probabilities = probabilities
        return

    def probability(self, features: torch.Tensor, /) -> torch.Tensor:
        _not_used = features
        return self.probabilities


class TestConditionedSplit:
    """Test the conditions of a split and its samples."""

    @pytest.fixture
    def split(self) -> ConditionedSplit:
        """Provide three subjects that the classifier gives 0.2."""
        split = DataSplit(torch.arange(6.0).view(3, 2), torch.ones(3, 1), subject_ids=('a', 'b', 'c'))
        return ConditionedSplit(split, _Classifier(torch.full((3, 1), 0.2)))

    def test_conditions(self, example_config: ExperimentConfig, split: ConditionedSplit) -> None:
        """The conditions are the probabilities of the classifier, rescaled as configured."""
        cfg_classification = example_config.classification
        steps = cfg_classification.rescale_steps[cfg_classification.model.name]

        expected = rescale(torch.full((3, 1), 0.2), steps)

        torch.testing.assert_close(split.conditions, expected)
        torch.testing.assert_close(split.probabilities, torch.full((3, 1), 0.2))
        assert split.subject_ids == ('a', 'b', 'c')

    def test_samples_carry_conditions(self, split: ConditionedSplit) -> None:
        """A sample gives the condition to the model."""
        inputs, targets = split[1]

        torch.testing.assert_close(inputs.features, torch.tensor([2.0, 3.0]))
        torch.testing.assert_close(inputs.condition, split.conditions[1])
        torch.testing.assert_close(targets.label, torch.tensor([1.0]))


class TestCounterfactualSplit:
    """Test the counterfactuals of a split."""

    split: ConditionedSplit
    module: CounterfactualVAE
    classifier: _Classifier

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """Set up two conditioned subjects, the VAE and a classifier answering 0.9."""
        original = DataSplit(torch.rand(2, N_FEATURES) * 2 - 1, torch.tensor([[0.0], [1.0]]), subject_ids=('a', 'b'))
        self.split = ConditionedSplit(original, _Classifier(torch.tensor([[0.2], [0.8]])))
        self.module = CounterfactualVAE().eval()
        self.classifier = _Classifier(torch.full((2, 1), 0.9))
        return

    def test_counterfactuals(self) -> None:
        """The counterfactuals are the reconstructions under the target, plus the residual when denoising."""
        targets = get_targets(self.split.conditions, CounterfactualTargets.HALF)
        denoised, _, recon_target = self.module.counterfactual(self.split.features, self.split.conditions, targets)

        noisy_split = CounterfactualSplit(
            self.split, self.module, self.classifier, denoise=False, target=CounterfactualTargets.HALF
        )
        denoised_split = CounterfactualSplit(
            self.split, self.module, self.classifier, denoise=True, target=CounterfactualTargets.HALF
        )

        torch.testing.assert_close(noisy_split.features, torch.clip(recon_target, min=-1, max=1))
        torch.testing.assert_close(denoised_split.features, denoised)
        torch.testing.assert_close(denoised_split.targets, targets)

    def test_keeps_the_original_subjects(self) -> None:
        """The counterfactuals keep the labels and conditions of the originals, with their own probabilities."""
        counterfactuals = CounterfactualSplit(
            self.split, self.module, self.classifier, denoise=True, target=CounterfactualTargets.OPPOSITE
        )

        torch.testing.assert_close(counterfactuals.labels, self.split.labels)
        torch.testing.assert_close(counterfactuals.conditions, self.split.conditions)
        torch.testing.assert_close(counterfactuals.probabilities, torch.full((2, 1), 0.9))
        assert counterfactuals.subject_ids == self.split.subject_ids


class TestReconstructedSplit:
    """Test the reconstructions of a split."""

    def test_reconstructions(self) -> None:
        """The features are the reconstructions under the conditions, which the split keeps."""
        original = DataSplit(torch.rand(2, N_FEATURES) * 2 - 1, torch.zeros(2, 1))
        split = ConditionedSplit(original, _Classifier(torch.tensor([[0.2], [0.8]])))
        module = CounterfactualVAE().eval()
        expected = module(Inputs(features=split.features, condition=split.conditions)).recon

        reconstructed = ReconstructedSplit(split, module, _Classifier(torch.full((2, 1), 0.6)))

        torch.testing.assert_close(reconstructed.features, expected)
        torch.testing.assert_close(reconstructed.conditions, split.conditions)
        torch.testing.assert_close(reconstructed.probabilities, torch.full((2, 1), 0.6))


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


def test_get_targets() -> None:
    """The opposite target is the class the classifier does not predict; half is 0.5 for all."""
    conditions = torch.tensor([[0.2], [0.7], [0.5]])

    opposite = get_targets(conditions, CounterfactualTargets.OPPOSITE)
    half = get_targets(conditions, CounterfactualTargets.HALF)

    torch.testing.assert_close(opposite, torch.tensor([[1.0], [0.0], [0.0]]))
    torch.testing.assert_close(half, torch.full((3, 1), 0.5))
