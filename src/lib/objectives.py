"""Losses and metrics of the classifiers, the autoencoder and the counterfactual VAE."""

from __future__ import annotations

from typing import ClassVar, override

import torch
import torch.nn.functional as F
from torch import Tensor

from drytorch.lib.accumulators import AbstractAccumulator
from drytorch.lib.objectives import DictLoss, Loss, LossWithMetrics, Metric, Objective, join

from src.config.experiment import Experiment
from src.data.structures import Outputs, Targets

__all__ = [
    'BinaryClassMetrics',
    'ConfusionMatrixAccumulator',
    'get_accuracy',
    'get_autoencoder_loss',
    'get_classifier_loss',
    'get_kld',
    'get_mse',
    'get_vae_loss',
]


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def get_accuracy() -> Metric[Tensor, Targets]:
    """Get the per-sample accuracy of the predicted diagnosis."""

    def _accuracy(logits: Tensor, targets: Targets) -> Tensor:
        return ((logits > 0) == targets.label).float().squeeze(1)

    return Metric(_accuracy, name='Accuracy', higher_is_better=True)


def get_classifier_loss() -> LossWithMetrics[Tensor, Targets]:
    """Get the binary cross-entropy, reporting the accuracy as well."""

    def _cross_entropy(logits: Tensor, targets: Targets) -> Tensor:
        return F.binary_cross_entropy_with_logits(logits, targets.label, reduction='none').squeeze(1)

    return join(Loss(_cross_entropy, name='CE'), get_accuracy())


class ConfusionMatrixAccumulator(AbstractAccumulator[Tensor, Tensor]):
    """Accumulator of confusion matrices.

    Attributes:
        matrix: Total counts by label (rows) and prediction (columns).
    """

    matrix: Tensor

    def __init__(self, matrix: Tensor) -> None:
        """Initialize.

        Args:
            matrix: counts of a batch, with labels on the rows.
        """
        self.matrix = matrix
        return

    @classmethod
    @override
    def from_value(cls, value: Tensor) -> ConfusionMatrixAccumulator:
        return cls(value.clone())

    @override
    def merge(self, other: ConfusionMatrixAccumulator) -> None:
        self.matrix += other.matrix
        return

    @override
    def sync(self) -> None:
        return

    @override
    def reduce(self) -> Tensor:
        return self.matrix.float()


class BinaryClassMetrics(Objective[Tensor, Targets]):
    """Accuracy, specificity and sensitivity over a whole partition.

    Typical development is the positive class.
    """

    _KEY: ClassVar[str] = 'confusion_matrix'
    _matrix: Tensor | None

    def __init__(self) -> None:
        """Initialize."""
        super().__init__()
        self._matrix = None
        return

    @property
    def confusion_matrix(self) -> Tensor:
        """Counts of the last computed evaluation, labels on the rows.

        Raises:
            ValueError: if the metrics have not been computed yet.
        """
        if self._matrix is None:
            msg = 'The confusion matrix is available after computing the metrics.'
            raise ValueError(msg)

        return self._matrix

    @override
    def calculate(self, outputs: Tensor, targets: Targets) -> dict[str, Tensor]:
        """Count the predictions of a batch for each label.

        Args:
            outputs: logits from the classifier.
            targets: labels of the batch.

        Returns:
            The confusion matrix of the batch with labels on the rows.
        """
        indices = 2 * targets.label.long() + (outputs > 0).long()
        return {self._KEY: torch.bincount(indices.flatten(), minlength=4).view(2, 2)}

    @classmethod
    @override
    def _get_accumulator_cls(cls) -> type[ConfusionMatrixAccumulator]:
        return ConfusionMatrixAccumulator

    @override
    def _compute(self) -> dict[str, Tensor]:
        if self._KEY not in self._aggregator.accumulators:
            return {}

        self._matrix = self._aggregator.accumulators[self._KEY].reduce()
        return self.get_metrics(self._matrix)

    @staticmethod
    def get_metrics(matrix: Tensor, /) -> dict[str, Tensor]:
        """Get accuracy, specificity and sensitivity from a confusion matrix.

        Args:
            matrix: the counts ``(2, 2)``, labels on the rows.

        Returns:
            The metrics, by name.
        """
        (tn, fp), (fn, tp) = matrix
        return {
            'Accuracy': (tn + tp) / matrix.sum(),
            'Specificity': tn / (tn + fp).clamp(min=1),
            'Sensitivity': tp / (tp + fn).clamp(min=1),
        }


# ---------------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------------


def get_mse() -> Loss[Tensor, Targets]:
    """Get the per-sample mean squared error of the reconstructed connections."""

    def _mse(recon: Tensor, targets: Targets) -> Tensor:
        return _get_mse(recon, targets.features)

    return Loss(_mse, name='MSE')


def get_autoencoder_loss() -> LossWithMetrics[Tensor, Targets]:
    """Get the loss of the autoencoder."""
    return get_mse()


def get_kld() -> DictLoss[Outputs, Targets]:
    """Get the KL divergence of the posterior from the prior, summed over the latent stages.

    The loss smooths the divergence of every latent coordinate with a softplus, which keeps
    it positive under sampling noise; ``KLD`` reports the divergence itself.
    """

    def _kld(outputs: Outputs, _: Targets) -> dict[str, Tensor]:
        mu, *d_mus = outputs.mu
        log_var, *d_log_vars = outputs.log_var
        matrices = [-1 - log_var + (mu - outputs.prior_mu) ** 2 + log_var.exp()]
        for d_mu, d_log_var, prior_log_var in zip(d_mus, d_log_vars, outputs.prior_log_var, strict=True):
            matrices.append(-1 - d_log_var + d_log_var.exp() + d_mu**2 / prior_log_var.exp())

        kld_smooth = sum(0.5 * F.softplus(matrix).sum(1) for matrix in matrices)
        kld = sum(0.5 * matrix.sum(1) for matrix in matrices)
        return {'KLD_smooth': torch.as_tensor(kld_smooth), 'KLD': torch.as_tensor(kld)}

    return DictLoss(_kld, name='KLD_smooth')


def get_vae_loss() -> LossWithMetrics[Outputs, Targets]:
    """Get the loss of the counterfactual VAE: reconstruction error and weighted KL divergence."""
    c_kld = Experiment.get_config().counterfactual.objective.c_kld

    def _mse(outputs: Outputs, targets: Targets) -> Tensor:
        return _get_mse(outputs.recon, targets.features)

    return Loss(_mse, name='MSE') + c_kld * get_kld()


def _get_mse(recon: Tensor, features: Tensor) -> Tensor:
    return F.mse_loss(recon, features, reduction='none').mean(1)
