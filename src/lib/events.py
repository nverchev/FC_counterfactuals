"""Events published during training and evaluation."""

from __future__ import annotations

import dataclasses

from matplotlib.figure import Figure
from torch import Tensor

from drytorch.core import log_events

__all__ = ['ConfusionMatrixEvent', 'FigureEvent']


@dataclasses.dataclass(frozen=True)
class ConfusionMatrixEvent(log_events.Event):
    """Event logged when an evaluation computes a confusion matrix.

    Attributes:
        source_name: Name of the runner that evaluated the model.
        model_name: Name of the evaluated model.
        epoch: Epoch of the evaluated weights.
        matrix: Counts of shape ``(n_classes, n_classes)``, labels on the rows.
    """

    source_name: str
    model_name: str
    epoch: int
    matrix: Tensor


@dataclasses.dataclass(frozen=True)
class FigureEvent(log_events.Event):
    """Figure to save, published by the scripts that draw them.

    Attributes:
        figure: the figure; the publisher closes it after the event.
        title: the name of the figure, which names its file.
        folder: the subfolder of the run's figures, empty for none.
    """

    figure: Figure
    title: str
    folder: str = ''
