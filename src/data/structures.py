"""Typed I/O structures for the models, the losses and the metrics."""

from __future__ import annotations

import dataclasses
from typing import NamedTuple

import torch

__all__ = [
    'Inputs',
    'Outputs',
    'Targets',
]


class Inputs(NamedTuple):
    """Inputs of the models.

    Attributes:
        features: Connections above the diagonal of the connectivity matrices ``(B, n_features)``.
        condition: Probability of the conditioning classifier ``(B, 1)`` (only for conditional autoencoders).
    """

    features: torch.Tensor
    condition: torch.Tensor


class Targets(NamedTuple):
    """Targets of the losses and the metrics.

    Attributes:
        label: Diagnosis ``(B, 1)``: 0 for autism spectrum disorder, 1 for typical development.
        features: The input features, which the autoencoders reconstruct ``(B, n_features)``.
    """

    label: torch.Tensor
    features: torch.Tensor


@dataclasses.dataclass(slots=True)
class Outputs:
    """Outputs of the counterfactual VAE.

    Attributes:
        recon: Reconstructed features ``(B, n_features)``.
        prior_mu: Mean of the prior of the top latent stage, set by the condition.
        mu: Means of the posterior: of the top latent stage, then the shifts from the prior of
            the stages below.
        log_var: Log-variances of the posterior: of the top latent stage, then the shifts from
            the prior of the stages below.
        prior_log_var: Log-variances of the prior of the stages below the top one.
    """

    recon: torch.Tensor
    prior_mu: torch.Tensor
    mu: list[torch.Tensor] = dataclasses.field(default_factory=list)
    log_var: list[torch.Tensor] = dataclasses.field(default_factory=list)
    prior_log_var: list[torch.Tensor] = dataclasses.field(default_factory=list)
