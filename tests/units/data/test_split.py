"""Tests for the dataset of a partition."""

import torch

from src.data.split import DataSplit


class TestDataSplit:
    """Test the samples of a split."""

    def test_samples_carry_features_and_labels(self) -> None:
        """A sample gives the features to the model and the loss, and an empty condition."""
        split = DataSplit(torch.arange(6.0).view(3, 2), torch.ones(3, 1))

        inputs, targets = split[1]

        torch.testing.assert_close(inputs.features, torch.tensor([2.0, 3.0]))
        assert inputs.condition.numel() == 0
        torch.testing.assert_close(targets.features, inputs.features)
        torch.testing.assert_close(targets.label, torch.tensor([1.0]))
