"""Partitions of the dataset and the dataset of a partition."""

from __future__ import annotations

import enum
from typing import Final

import torch
from torch.utils.data import Dataset

from src.data.structures import Inputs, Targets

__all__ = [
    'DataSplit',
    'Partitions',
]


class Partitions(enum.Enum):
    """Named dataset splits."""

    train = enum.auto()
    train_val = enum.auto()
    val = enum.auto()
    test = enum.auto()


class DataSplit(Dataset[tuple[Inputs, Targets]]):
    """Subjects of a partition, with the probability that conditions the counterfactual VAE.

    Attributes:
        features: Connections above the diagonal of the connectivity matrices.
        labels: Diagnoses, 0 for autism spectrum disorder and 1 for typical development.
        conditions: Probabilities of the conditioning classifier, zeros until set.
        subject_ids: Identifiers of the subjects, as the ABIDE files name them.
    """

    features: Final[torch.Tensor]
    labels: Final[torch.Tensor]
    conditions: Final[torch.Tensor]
    subject_ids: Final[tuple[str, ...]]

    def __init__(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        *,
        conditions: torch.Tensor | None = None,
        subject_ids: tuple[str, ...] = (),
    ) -> None:
        """Initialize.

        Args:
            features: connections above the diagonal of the connectivity matrices ``(N, n_features)``.
            labels: diagnoses ``(N, 1)``.
            conditions: probabilities of the conditioning classifier ``(N, 1)``; zeros when omitted.
            subject_ids: identifiers of the subjects.
        """
        super().__init__()
        self.features = features
        self.labels = labels
        self.conditions = torch.zeros_like(labels) if conditions is None else conditions
        self.subject_ids = subject_ids
        return

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> tuple[Inputs, Targets]:
        features = self.features[index]
        inputs = Inputs(features=features, condition=self.conditions[index])
        return inputs, Targets(label=self.labels[index], features=features)

    def with_conditions(self, conditions: torch.Tensor, /) -> DataSplit:
        """Get the same subjects with the given conditioning probabilities.

        Args:
            conditions: probabilities of the conditioning classifier ``(N, 1)``.

        Returns:
            A new split sharing the features and the labels.
        """
        return DataSplit(self.features, self.labels, conditions=conditions, subject_ids=self.subject_ids)
