"""Weights & Biases tracker handling the events of this project."""

import functools

from typing import Any, override

import wandb

from drytorch.core import log_events
from drytorch.trackers.wandb import Wandb

from src.lib.events import ConfusionMatrixEvent, FigureEvent

__all__ = ['ClassificationWandb']


class ClassificationWandb(Wandb):
    """Weights & Biases tracker that also logs confusion matrices and figures."""

    @functools.singledispatchmethod
    @override
    def notify(self, event: log_events.Event) -> None:
        return super().notify(event)

    @notify.register
    def _(self, event: ConfusionMatrixEvent) -> None:
        counts = event.matrix.round().long().tolist()
        columns: list[Any] = ['label', *(f'predicted_{i}' for i in range(len(counts)))]
        rows: list[Any] = [[str(label), *row] for label, row in enumerate(counts)]
        table = wandb.Table(columns=columns, data=rows)
        name = f'{event.model_name}/{event.source_name}-confusion_matrix'
        self.log_at_epoch({name: table}, model_name=event.model_name, epoch=event.epoch)
        return super().notify(event)

    @notify.register
    def _(self, event: FigureEvent) -> None:
        if self.record:
            name = f'{event.folder}/{event.title}' if event.folder else event.title
            self.run.log({name: wandb.Image(event.figure)})

        return super().notify(event)
