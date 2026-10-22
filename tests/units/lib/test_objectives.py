"""Tests for the losses and the metrics."""

import math

import pytest
import torch

from src.data.structures import Outputs, Targets
from src.lib.objectives import BinaryClassMetrics, get_accuracy, get_kld, get_mse


def _get_targets(labels: list[float]) -> Targets:
    label = torch.tensor(labels).unsqueeze(1)
    return Targets(label=label, features=torch.zeros(len(labels), 3))


def test_get_accuracy() -> None:
    """A positive logit predicts typical development."""
    logits = torch.tensor([[2.0], [-1.0], [0.5]])

    accuracy = get_accuracy().calculate(logits, _get_targets([1, 1, 0]))

    torch.testing.assert_close(accuracy['Accuracy'], torch.tensor([1.0, 0.0, 0.0]))


def test_get_mse() -> None:
    """The error is averaged over the connections of each subject."""
    recon = torch.tensor([[1.0, 1.0, 1.0], [2.0, 0.0, 0.0]])
    targets = Targets(label=torch.zeros(2, 1), features=torch.zeros(2, 3))

    mse = get_mse().calculate(recon, targets)

    torch.testing.assert_close(mse['MSE'], torch.tensor([1.0, 4 / 3]))


class TestKLD:
    """Test the KL divergence of the posterior from the prior."""

    @staticmethod
    def _get_outputs(top_mu: torch.Tensor, prior_mu: torch.Tensor) -> Outputs:
        zeros = torch.zeros_like(top_mu)
        return Outputs(zeros, prior_mu, mu=[top_mu, zeros], log_var=[zeros, zeros], prior_log_var=[zeros])

    def test_posterior_equal_to_the_prior(self) -> None:
        """The divergence is zero; its smooth version is the softplus of zero for each coordinate."""
        outputs = self._get_outputs(torch.full((2, 4), 0.3), torch.full((2, 4), 0.3))

        values = get_kld().calculate(outputs, _get_targets([0, 1]))

        torch.testing.assert_close(values['KLD'], torch.zeros(2))
        torch.testing.assert_close(values['KLD_smooth'], torch.full((2,), 0.5 * 8 * math.log(2)))

    def test_shifted_mean(self) -> None:
        """A unit shift of the top mean costs one half."""
        outputs = self._get_outputs(torch.tensor([[1.0, 0.0]]), torch.zeros(1, 2))

        values = get_kld().calculate(outputs, _get_targets([0]))

        torch.testing.assert_close(values['KLD'], torch.tensor([0.5]))


class TestBinaryClassMetrics:
    """Test the metrics computed over a whole partition."""

    @pytest.fixture
    def metrics(self) -> BinaryClassMetrics:
        """Provide the metrics after two batches."""
        metrics = BinaryClassMetrics()
        metrics.update(torch.tensor([[1.0], [1.0]]), _get_targets([1, 0]))
        metrics.update(torch.tensor([[-1.0], [-1.0], [1.0]]), _get_targets([0, 0, 1]))
        return metrics

    def test_metrics(self, metrics: BinaryClassMetrics) -> None:
        """Typical development is the positive class."""
        values = metrics.compute()

        torch.testing.assert_close(values['Accuracy'], torch.tensor(0.8))
        torch.testing.assert_close(values['Specificity'], torch.tensor(2 / 3))
        torch.testing.assert_close(values['Sensitivity'], torch.tensor(1.0))

    def test_confusion_matrix(self, metrics: BinaryClassMetrics) -> None:
        """The counts have the labels on the rows."""
        metrics.compute()

        matrix = metrics.confusion_matrix

        torch.testing.assert_close(matrix, torch.tensor([[2.0, 1.0], [0.0, 2.0]]))
