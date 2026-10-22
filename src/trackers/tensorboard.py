"""TensorBoard tracker handling the events of this project."""

import functools

from typing import override

from torch import Tensor

from drytorch.core import log_events
from drytorch.trackers.tensorboard import TensorBoard

from src.lib.events import ConfusionMatrixEvent, FigureEvent

__all__ = ['ClassificationTensorBoard']


class ClassificationTensorBoard(TensorBoard):
    """TensorBoard tracker that also writes confusion matrices and figures."""

    @functools.singledispatchmethod
    @override
    def notify(self, event: log_events.Event) -> None:
        return super().notify(event)

    @notify.register
    def _(self, event: ConfusionMatrixEvent) -> None:
        name = f'{event.model_name}/{event.source_name}-confusion_matrix'
        # each row is scaled by its support, so the image reads as accuracies
        shares = event.matrix / event.matrix.sum(dim=1, keepdim=True).clamp(min=1)
        self.writer.add_image(name, shares.unsqueeze(0), global_step=event.epoch)
        self.writer.add_text(name, self._markdown(event.matrix), global_step=event.epoch)
        return super().notify(event)

    @notify.register
    def _(self, event: FigureEvent) -> None:
        if self.record:
            name = f'{event.folder}/{event.title}' if event.folder else event.title
            self.writer.add_figure(name, event.figure, close=False)

        return super().notify(event)

    @staticmethod
    def _markdown(matrix: Tensor) -> str:
        """Return the counts as a markdown table, labels on the rows."""
        counts: list[list[int]] = matrix.round().long().tolist()
        names = [str(i) for i in range(len(counts))]
        header = '| label \\ predicted | ' + ' | '.join(names) + ' |'
        separator = '|---' * (len(names) + 1) + '|'
        rows = [
            f'| {name} | ' + ' | '.join(str(count) for count in row) + ' |'
            for name, row in zip(names, counts, strict=True)
        ]
        return '\n'.join([header, separator, *rows])
