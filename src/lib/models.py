"""Models of the experiment and the loading of the trained ones."""

import logging
from collections.abc import Callable
from typing import Any

import torch

from drytorch.core.exceptions import ModelNotFoundError
from drytorch.lib.models import Model

from src.config.experiment import Experiment
from src.config.options import Classifiers
from src.data.structures import Inputs, Outputs
from src.utils.svc import load_svc_probability
from src.module import Autoencoder, CounterfactualVAE, MLPClassifier

__all__ = [
    'AutoencoderModel',
    'ClassifierModel',
    'CounterfactualModel',
    'get_autoencoder_model',
    'get_counterfactual_model',
    'get_mlp_model',
    'load_classifier_probability',
    'load_pretrained_encoder',
    'load_trained_state',
]


class AutoencoderModel(Model[Inputs, torch.Tensor]):
    """Model wrapping the autoencoder that pretrains the encoder of the ``ae`` classifier.

    Attributes:
        module: the autoencoder.
    """

    module: Autoencoder


class ClassifierModel(Model[Inputs, torch.Tensor]):
    """Model wrapping the MLP classifier.

    Attributes:
        module: the classifier.
    """

    module: MLPClassifier


class CounterfactualModel(Model[Inputs, Outputs]):
    """Model wrapping the counterfactual VAE.

    Attributes:
        module: the counterfactual VAE.
    """

    module: CounterfactualVAE


def get_autoencoder_model() -> AutoencoderModel:
    """Get the autoencoder that pretrains the encoder of the ``ae`` classifier."""
    cfg = Experiment.get_config()
    return AutoencoderModel(Autoencoder(), name=cfg.autoencoding.model.name, device=cfg.user.device)


def get_counterfactual_model() -> CounterfactualModel:
    """Get the counterfactual VAE."""
    cfg = Experiment.get_config()
    return CounterfactualModel(CounterfactualVAE(), name=cfg.counterfactual.model.name, device=cfg.user.device)


def get_mlp_model() -> ClassifierModel:
    """Get the MLP classifier, named after the configured classifier."""
    cfg = Experiment.get_config()
    return ClassifierModel(MLPClassifier(), name=cfg.classification.model.name, device=cfg.user.device)


def load_classifier_probability() -> Callable[[torch.Tensor], torch.Tensor]:
    """Load the trained classifier and get its probability of typical development.

    Returns:
        A function from the connections ``(N, n_features)`` to the probabilities ``(N, 1)``, on
        the same device.

    Raises:
        RuntimeError: if the experiment holds no trained classifier.
    """
    cfg = Experiment.get_config()
    if cfg.classification.model.name == Classifiers.SVC:
        return load_svc_probability()

    model = get_mlp_model()
    load_trained_state(model)
    module = model.module.eval()

    @torch.inference_mode()
    def _probability(features: torch.Tensor) -> torch.Tensor:
        inputs = Inputs(features=features.to(model.device), condition=torch.empty(0))
        return torch.sigmoid(module(inputs)).to(features.device)

    return _probability


def load_pretrained_encoder(classifier: MLPClassifier, /) -> None:
    """Copy the encoder of the trained autoencoder of the experiment into the classifier.

    Args:
        classifier: the classifier receiving the encoder.

    Side Effects:
        Replaces the weights of the encoder of the classifier.
    """
    model = get_autoencoder_model()
    load_trained_state(model)
    classifier.encoder.load_state_dict(model.module.encoder.state_dict())
    logging.info('Loaded the encoder of the autoencoder at epoch %d.', model.epoch)
    return


def load_trained_state(model: Model[Any, Any], /, *, epoch: int = -1) -> None:
    """Load the weights of a model trained by an earlier phase of the experiment.

    Args:
        model: the model to load.
        epoch: the epoch to load. Default loads the last saved epoch.

    Raises:
        RuntimeError: if the experiment holds no trained model.
    """
    try:
        model.load_state(epoch)
    except ModelNotFoundError as error:
        cfg = Experiment.get_config()
        msg = (
            f'No trained {model.name} for the experiment {cfg.name} in {cfg.par_dir}: '
            'train it first with the same configuration and overrides.'
        )
        raise RuntimeError(msg) from error

    return
