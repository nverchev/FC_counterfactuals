"""Evaluate the trained classifier on the evaluation partition.

The evaluation partition is the test set only with ``final=true``, the validation set otherwise.

Usage::

    uv run evaluate_classifier.py final=true
    uv run evaluate_classifier.py --config-name mlp final=true
    uv run evaluate_classifier.py --config-name ae final=true
"""

from typing import cast

import torch

from drytorch import Test
from drytorch.core import log_events

from src.config import ExperimentConfig, Experiment, hydra_main
from src.config.environment import DEBUG_MODE
from src.config.options import Classifiers
from src.data import DataSplit, get_splits
from src.lib import get_loaders
from src.lib.events import ConfusionMatrixEvent
from src.lib.models import get_mlp_model, load_trained_state
from src.lib.objectives import BinaryClassMetrics
from src.trackers import get_trackers
from src.utils.svc import get_svc_confusion_matrix, load_svc


def evaluate_mlp(eval_split: DataSplit) -> tuple[torch.Tensor, int, str]:
    """Test the MLP classifier and get its confusion matrix.

    Args:
        eval_split: split of the evaluation partition.

    Returns:
        The confusion matrix on the evaluation partition, the epoch of the evaluated weights, and the name of the
        model.
    """
    cfg = Experiment.get_config()
    model = get_mlp_model()
    load_trained_state(model)
    _, eval_loader = get_loaders(eval_split, eval_split, batch_size=cfg.classification.train.batch_size)
    test = Test(model, loader=eval_loader, metric=BinaryClassMetrics(), name='Classification')
    test()
    return cast(BinaryClassMetrics, test.objective).confusion_matrix, model.epoch, model.name


def evaluate_svc(eval_split: DataSplit) -> tuple[torch.Tensor, int, str]:
    """Publish the metrics of the support vector classifier, as the MLP's test does, and get its confusion matrix.

    Args:
        eval_split: split of the evaluation partition.

    Returns:
        The confusion matrix on the evaluation partition, the epoch, one for a fitted classifier, and the name
        of the classifier.
    """
    matrix = get_svc_confusion_matrix(load_svc(), eval_split)
    name = str(Classifiers.SVC)
    metrics = {key: value.item() for key, value in BinaryClassMetrics.get_metrics(matrix.float()).items()}
    log_events.MetricEvent(model_name=name, source_name='Classification', epoch=1, metrics=metrics)
    return matrix, 1, name


def evaluate_classifier() -> None:
    """Evaluate the configured classifier.

    Side Effects:
        Publishes the confusion matrix of the classifier on the evaluation partition.
    """
    cfg = Experiment.get_config()
    classifier = cfg.classification.model.name
    _, eval_split = get_splits()
    if classifier == Classifiers.SVC:
        matrix, epoch, name = evaluate_svc(eval_split)
    else:
        matrix, epoch, name = evaluate_mlp(eval_split)

    ConfusionMatrixEvent(source_name='Classification', model_name=name, epoch=epoch, matrix=matrix)
    return


@hydra_main
def main(cfg: ExperimentConfig) -> None:
    """Set up the experiment and evaluate the classifier."""
    trackers = get_trackers(cfg)
    exp = Experiment(cfg, name=cfg.name, par_dir=cfg.par_dir, tags=cfg.tags)
    for tracker in trackers:
        exp.trackers.subscribe(tracker)

    with exp.create_run(resume=True, run_id=cfg.user.run_id, record=not DEBUG_MODE):
        evaluate_classifier()

    return


if __name__ == '__main__':
    main()
