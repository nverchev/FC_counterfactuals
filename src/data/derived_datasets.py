"""Splits carrying the data that the trained models derive from the subjects."""

from __future__ import annotations

import abc
import math
from typing import Final, Protocol, override

import torch

from src.config.experiment import Experiment
from src.config.options import Classifiers, CounterfactualTargets
from src.data.split import DataSplit
from src.data.structures import Inputs, Targets
from src.lib.models import ClassifierModel, get_mlp_model, load_trained_state
from src.module import CounterfactualVAE
from src.sklearn.svc import SVCClassifier, load_svc

__all__ = [
    'ConditionedSplit',
    'CounterfactualSplit',
    'ReconstructedSplit',
    'get_targets',
    'load_classifier',
    'rescale',
]


class _ClassifiedSplit(DataSplit):
    conditions: Final[torch.Tensor]
    probabilities: Final[torch.Tensor]

    def __init__(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        *,
        conditions: torch.Tensor,
        probabilities: torch.Tensor,
        subject_ids: tuple[str, ...],
    ) -> None:
        super().__init__(features, labels, subject_ids=subject_ids)
        self.conditions = conditions
        self.probabilities = probabilities
        return

    @override
    def __getitem__(self, index: int) -> tuple[Inputs, Targets]:
        features = self.features[index]
        inputs = Inputs(features=features, condition=self.conditions[index])
        return inputs, Targets(label=self.labels[index], features=features)


class ConditionedSplit(_ClassifiedSplit):
    """Subjects of a partition with the rescaled probabilities of the classifier as conditions.

    Attributes:
        conditions: Rescaled probabilities of the classifier ``(N, 1)``.
        probabilities: Probabilities of the classifier ``(N, 1)``.
    """

    def __init__(self, split: DataSplit, classifier: _ClassifierProtocol) -> None:
        """Initialize.

        Args:
            split: the split to condition.
            classifier: the trained classifier.
        """
        cfg_classification = Experiment.get_config().classification
        steps = cfg_classification.rescale_steps[cfg_classification.model.name]
        probabilities = classifier.probability(split.features)
        super().__init__(
            split.features,
            split.labels,
            conditions=rescale(probabilities, steps),
            probabilities=probabilities,
            subject_ids=split.subject_ids,
        )
        return


class CounterfactualSplit(_ClassifiedSplit):
    """Counterfactuals of the subjects of a conditioned split.

    The counterfactuals are the reconstructions under the target. Denoising removes their
    reconstruction error by adding the residual, the part of the original that the
    reconstruction under the condition misses.

    Attributes:
        conditions: Conditions of the original subjects ``(N, 1)``.
        probabilities: Probabilities the classifier assigns to the counterfactuals ``(N, 1)``.
        targets: Probabilities the counterfactuals aim at ``(N, 1)``.
    """

    targets: Final[torch.Tensor]

    def __init__(
        self,
        split: ConditionedSplit,
        module: CounterfactualVAE,
        classifier: _ClassifierProtocol,
        *,
        denoise: bool,
        target: CounterfactualTargets,
    ) -> None:
        """Initialize.

        Args:
            split: the original subjects.
            module: the trained counterfactual VAE.
            classifier: the trained classifier.
            denoise: whether to remove the reconstruction error from the counterfactuals.
            target: the strategy choosing the probabilities the counterfactuals aim at.
        """
        targets = get_targets(split.conditions, target)
        device = next(module.parameters()).device
        features, conditions = split.features.to(device), split.conditions.to(device)
        with torch.inference_mode():
            denoised, _, recon_target = module.eval().counterfactual(features, conditions, targets.to(device))

        counterfactuals = denoised if denoise else torch.clip(recon_target, min=-1, max=1)
        super().__init__(
            counterfactuals.cpu(),
            split.labels,
            conditions=split.conditions,
            probabilities=classifier.probability(counterfactuals).cpu(),
            subject_ids=split.subject_ids,
        )
        self.targets = targets
        return


class ReconstructedSplit(_ClassifiedSplit):
    """Reconstructions of the subjects of a conditioned split under their conditions.

    Attributes:
        conditions: Conditions of the original subjects ``(N, 1)``.
        probabilities: Probabilities the classifier assigns to the reconstructions ``(N, 1)``.
    """

    def __init__(self, split: ConditionedSplit, module: CounterfactualVAE, classifier: _ClassifierProtocol) -> None:
        """Initialize.

        Args:
            split: the original subjects.
            module: the trained counterfactual VAE.
            classifier: the trained classifier.
        """
        device = next(module.parameters()).device
        inputs = Inputs(features=split.features.to(device), condition=split.conditions.to(device))
        with torch.inference_mode():
            recon = module.eval()(inputs).recon

        super().__init__(
            recon.cpu(),
            split.labels,
            conditions=split.conditions,
            probabilities=classifier.probability(recon).cpu(),
            subject_ids=split.subject_ids,
        )
        return


def get_targets(conditions: torch.Tensor, target: CounterfactualTargets) -> torch.Tensor:
    """Get the probabilities the counterfactuals aim at.

    Args:
        conditions: probabilities of the conditioning classifier.
        target: the strategy choosing the targets.

    Returns:
        The opposite class (0 or 1) of each subject, or 0.5 for all.
    """
    if target == CounterfactualTargets.HALF:
        return torch.full_like(conditions, 0.5)

    return (conditions < 0.5).to(conditions.dtype)


def load_classifier() -> ClassifierModel | SVCClassifier:
    """Load the trained classifier whose probabilities condition the counterfactual VAE.

    Returns:
        The MLP classifier or the support vector classifier, as configured.

    Raises:
        RuntimeError: if the experiment holds no trained classifier.
    """
    cfg = Experiment.get_config()
    if cfg.classification.model.name == Classifiers.SVC:
        return SVCClassifier(load_svc())

    model = get_mlp_model()
    load_trained_state(model)
    return model


def rescale(probabilities: torch.Tensor, steps: int) -> torch.Tensor:
    """Spread probabilities close to 0 or 1 towards the middle.

    Each step applies ``arccos(1 - 2p) / pi``, which keeps 0, 0.5 and 1 and the order of the
    probabilities, so that a saturated classifier still gives the VAE a usable condition.

    Args:
        probabilities: probabilities in [0, 1].
        steps: number of times the map is applied.

    Returns:
        The rescaled probabilities, with the same type as the input.
    """
    rescaled = probabilities.double()
    for _ in range(steps):
        rescaled = torch.arccos(1 - 2 * rescaled) / math.pi

    return rescaled.to(probabilities.dtype)


class _ClassifierProtocol(Protocol):
    @abc.abstractmethod
    def probability(self, features: torch.Tensor, /) -> torch.Tensor:
        """Get the probability of typical development of each subject."""
