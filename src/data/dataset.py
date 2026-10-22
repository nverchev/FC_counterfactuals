"""Dataset factory for the training and evaluation partitions."""

from src.config.experiment import Experiment
from src.config.options import Datasets
from src.data.abide import AbideDataBuilder
from src.data.databuilder import DataBuilder
from src.data.split import DataSplit, Partitions

__all__ = [
    'get_databuilder',
    'get_partitions',
    'get_splits',
]


def get_databuilder(name: Datasets | None = None, /) -> DataBuilder:
    """Return the dataset builder instance corresponding to dataset name.

    Args:
        name: dataset enum identifier; the configured one when omitted.

    Returns:
        The dataset builder, built once for each data configuration.
    """
    if name is None:
        name = Experiment.get_config().data.dataset.name

    databuilder_dict: dict[Datasets, type[DataBuilder]] = {
        Datasets.ABIDE: AbideDataBuilder,
    }
    return databuilder_dict[name]()


def get_partitions() -> tuple[Partitions, Partitions]:
    """Get the training and the evaluation partitions of the current experiment.

    Returns:
        ``train`` and ``val``, or ``train_val`` and ``test`` when ``final`` is set.
    """
    if Experiment.get_config().final:
        return Partitions.train_val, Partitions.test

    return Partitions.train, Partitions.val


def get_splits() -> tuple[DataSplit, DataSplit]:
    """Get the training and the evaluation splits of the current experiment.

    Returns:
        The splits of the partitions of :func:`get_partitions`.
    """
    databuilder = get_databuilder()
    train_partition, eval_partition = get_partitions()
    return databuilder.split(train_partition), databuilder.split(eval_partition)
