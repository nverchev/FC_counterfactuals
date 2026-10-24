"""Support vector classifier: fitting, saving, loading and probabilities."""

import logging
import pathlib

from typing import Final, cast

import joblib
import torch

from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC

from src.config.experiment import Experiment
from src.config.options import Classifiers
from src.data.split import DataSplit

__all__ = [
    'SVCClassifier',
    'fit_and_save_svc',
    'get_svc_confusion_matrix',
    'get_svc_path',
    'load_svc',
]


class SVCClassifier:
    """Fitted support vector classifier giving the probability of typical development.

    Attributes:
        svc: The fitted classifier.
    """

    svc: Final[CalibratedClassifierCV]

    def __init__(self, svc: CalibratedClassifierCV, /) -> None:
        """Initialize.

        Args:
            svc: the fitted classifier.
        """
        self.svc = svc
        return

    def probability(self, features: torch.Tensor, /) -> torch.Tensor:
        """Get the probability of typical development.

        Args:
            features: connections ``(N, n_features)``.

        Returns:
            The probabilities ``(N, 1)``, on the device of the features.
        """
        probabilities = self.svc.predict_proba(features.cpu().double().numpy())[:, 1]
        return torch.from_numpy(probabilities).float().unsqueeze(1).to(features.device)


def fit_and_save_svc(split: DataSplit, /) -> CalibratedClassifierCV:
    """Fit the support vector classifier and save it in the folder of the experiment.

    Its probabilities come from a sigmoid fitted on cross-validated decision values.

    Args:
        split: split of the training partition.

    Returns:
        The fitted classifier.

    Raises:
        ValueError: if the configured classifier is not the support vector classifier.

    Side Effects:
        Saves the classifier at :func:`get_svc_path`.
    """
    cfg_svc = Experiment.get_config().classification.model
    if cfg_svc.name != Classifiers.SVC:
        raise ValueError('The configured classifier is not the support vector classifier.')

    svc = CalibratedClassifierCV(SVC(kernel=cfg_svc.kernel, C=cfg_svc.c, gamma=cfg_svc.gamma), ensemble=False)
    logging.info('Fitting the support vector classifier.')
    svc.fit(split.features.double().numpy(), split.labels.squeeze(1).numpy())
    path = get_svc_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(svc, path)
    return svc


def get_svc_confusion_matrix(svc: CalibratedClassifierCV, split: DataSplit, /) -> torch.Tensor:
    """Get the confusion matrix of the classifier on a split.

    Args:
        svc: the fitted classifier.
        split: the split to classify.

    Returns:
        The counts ``(2, 2)``, labels on the rows.
    """
    predictions = torch.from_numpy(svc.predict(split.features.double().numpy())).long()
    labels = split.labels.squeeze(1).long()
    return torch.bincount(2 * labels + predictions, minlength=4).view(2, 2)


def get_svc_path() -> pathlib.Path:
    """Get the file of the support vector classifier of the experiment."""
    cfg = Experiment.get_config()
    return cfg.par_dir / 'checkpoints' / cfg.name / 'svc.joblib'


def load_svc() -> CalibratedClassifierCV:
    """Load the support vector classifier of the experiment.

    Returns:
        The fitted classifier.

    Raises:
        RuntimeError: if the experiment holds no trained support vector classifier.
    """
    path = get_svc_path()
    if not path.exists():
        msg = f'No trained support vector classifier at {path}: train it first.'
        raise RuntimeError(msg)

    return cast(CalibratedClassifierCV, joblib.load(path))
