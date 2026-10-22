"""Tests for the dataset of a partition."""

import torch

from src.data.split import DataSplit


class TestDataSplit:
    """Test the samples and the conditions of a split."""

    def test_samples_carry_features_conditions_and_labels(self) -> None:
        """A sample gives the features to the model and the loss, the condition to the model."""
        split = DataSplit(torch.arange(6.0).view(3, 2), torch.ones(3, 1), conditions=torch.full((3, 1), 0.3))

        inputs, targets = split[1]

        torch.testing.assert_close(inputs.features, torch.tensor([2.0, 3.0]))
        torch.testing.assert_close(inputs.condition, torch.tensor([0.3]))
        torch.testing.assert_close(targets.features, inputs.features)
        torch.testing.assert_close(targets.label, torch.tensor([1.0]))

    def test_conditions_are_zero_until_set(self) -> None:
        """A split without conditions has zeros, and setting them keeps the subjects."""
        split = DataSplit(torch.zeros(4, 2), torch.zeros(4, 1), subject_ids=('a', 'b', 'c', 'd'))

        conditioned = split.with_conditions(torch.full((4, 1), 0.7))

        torch.testing.assert_close(split.conditions, torch.zeros(4, 1))
        torch.testing.assert_close(conditioned.conditions, torch.full((4, 1), 0.7))
        assert conditioned.features is split.features
        assert conditioned.subject_ids == split.subject_ids
