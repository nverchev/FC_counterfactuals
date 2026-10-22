"""DataLoader factory for training and evaluation."""

from drytorch import DataLoader

from src.data.split import DataSplit

__all__ = ['get_loaders']


def get_loaders(
    train_split: DataSplit,
    eval_split: DataSplit,
    *,
    batch_size: int,
    n_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Get the loaders of the training and the evaluation datasets.

    Args:
        train_split: split of the training partition.
        eval_split: split of the evaluation partition.
        batch_size: number of samples per batch.
        n_workers: number of worker subprocesses.

    Returns:
        The training loader and the evaluation loader.
    """
    train_loader = DataLoader(train_split, batch_size=batch_size, n_workers=n_workers)
    eval_loader = DataLoader(eval_split, batch_size=batch_size, n_workers=n_workers)
    return train_loader, eval_loader
