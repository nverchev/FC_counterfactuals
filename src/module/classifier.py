"""MLP classifier, whose encoder the autoencoder pretrains."""

import abc
import itertools
from collections.abc import Sequence
from typing import Final

import torch
import torch.nn as nn

from src.data.abide import N_FEATURES
from src.config.experiment import Experiment
from src.config.options import Classifiers
from src.data.structures import Inputs

__all__ = [
    'BaseEncoderArchitecture',
    'MLPClassifier',
]


class BaseEncoderArchitecture(nn.Module, abc.ABC):
    """Abstract base for the models built on the encoder of the classifier.

    Attributes:
        in_dim: Connections of each connectivity matrix.
        hidden_dims: Widths of the hidden layers of the encoder.
        z_dim: Width of the encoding.
        encoder: Linear layers with ReLU activations between them, mapping the connections to
            the encoding.
    """

    in_dim: Final[int]
    hidden_dims: Final[tuple[int, ...]]
    z_dim: Final[int]
    encoder: Final[nn.Sequential]

    def __init__(self) -> None:
        """Initialize.

        Raises:
            ValueError: if the configured classifier is the support vector classifier.
        """
        super().__init__()
        cfg = Experiment.get_config()
        cfg_model = cfg.classification.model
        if cfg_model.name == Classifiers.SVC:
            raise ValueError('The support vector classifier is not a torch model.')

        self.in_dim = N_FEATURES
        self.hidden_dims = cfg_model.hidden_dims
        self.z_dim = cfg_model.z_dim
        self.encoder = self.get_perceptron([self.in_dim, *self.hidden_dims, self.z_dim])
        return

    @abc.abstractmethod
    def forward(self, inputs: Inputs, /) -> torch.Tensor:
        """Run the forward pass."""

    @staticmethod
    def get_perceptron(dims: Sequence[int], /) -> nn.Sequential:
        """Get linear layers with ReLU activations between them, and none after the last.

        Args:
            dims: the widths of the input and of the output of each layer.

        Returns:
            The layers in sequence.
        """
        layers = nn.Sequential()
        for in_dim, out_dim in itertools.pairwise(dims):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())

        return layers[:-1]


class MLPClassifier(BaseEncoderArchitecture):
    """MLP predicting typical development from the connections.

    Attributes:
        head: ReLU and linear layer mapping the encoding to the logit.
    """

    head: Final[nn.Sequential]

    def __init__(self) -> None:
        """Initialize."""
        super().__init__()
        self.head = nn.Sequential(nn.ReLU(), nn.Linear(self.z_dim, 1))
        return

    def forward(self, inputs: Inputs, /) -> torch.Tensor:
        """Get the logit of typical development."""
        return self.head(self.encoder(inputs.features))
