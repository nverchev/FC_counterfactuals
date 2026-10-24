"""Tests for the MLP classifier."""

import torch
import torch.nn as nn

from src.data.structures import Inputs
from src.module import autoencoders
from src.module.classifiers import BaseEncoderArchitecture, MLPClassifier

from src.data.abide import N_FEATURES


def _get_inputs(n_samples: int = 3) -> Inputs:
    return Inputs(features=torch.randn(n_samples, N_FEATURES), condition=torch.zeros(n_samples, 1))


def test_classifier_gives_one_logit_per_subject() -> None:
    """The classifier outputs the logit of typical development."""
    logits = MLPClassifier()(_get_inputs())

    assert logits.shape == (3, 1)


def test_classifier_takes_the_encoder_of_the_autoencoder() -> None:
    """The encoders match, so the weights of one load into the other."""
    autoencoder, classifier = autoencoders.Autoencoder(), MLPClassifier()

    classifier.encoder.load_state_dict(autoencoder.encoder.state_dict())

    for weight, pretrained in zip(classifier.encoder.parameters(), autoencoder.encoder.parameters(), strict=True):
        torch.testing.assert_close(weight, pretrained)


def test_get_perceptron() -> None:
    """Linear layers alternate with ReLU activations, with none after the last."""
    layers = BaseEncoderArchitecture.get_perceptron([5, 4, 3])

    types = [type(layer) for layer in layers]

    assert types == [nn.Linear, nn.ReLU, nn.Linear]
