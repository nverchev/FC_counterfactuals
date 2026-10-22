"""Autoencoder pretraining the classifier and hierarchical VAE generating counterfactual connectomes."""

from __future__ import annotations

import dataclasses
import itertools
from typing import Final, cast, override

import torch
import torch.nn as nn

from src.data.abide import N_FEATURES
from src.config.experiment import Experiment
from src.data.structures import Inputs, Outputs
from src.module.classifier import BaseEncoderArchitecture

__all__ = [
    'Autoencoder',
    'CounterfactualVAE',
    'Encoding',
    'LatentStage',
]


class Autoencoder(BaseEncoderArchitecture):
    """Autoencoder pretraining the encoder of the classifier.

    Attributes:
        decoder: Linear layers with ReLU activations between them, mirroring the encoder.
    """

    decoder: Final[nn.Sequential]

    def __init__(self) -> None:
        """Initialize."""
        super().__init__()
        self.decoder = self.get_perceptron([self.z_dim, *reversed(self.hidden_dims), self.in_dim])
        return

    def forward(self, inputs: Inputs, /) -> torch.Tensor:
        """Reconstruct the connections from their encoding."""
        return self.decoder(self.encoder(inputs.features))


@dataclasses.dataclass(frozen=True)
class Encoding:
    """Encoder outputs that the decoder reads.

    Attributes:
        z: Sample of the top latent stage.
        mu: Mean of the posterior of the top latent stage.
        log_var: Log-variance of the posterior of the top latent stage.
        hidden: Outputs of the hidden layers of the encoder, from the deepest.
    """

    z: torch.Tensor
    mu: torch.Tensor
    log_var: torch.Tensor
    hidden: tuple[torch.Tensor, ...]


class LatentStage(nn.Module):
    """Latent stage of the decoder, between two of its hidden layers.

    The prior of the stage depends on the latent above it; the posterior shifts the prior by
    an amount that the matching hidden layer of the encoder sets.

    Attributes:
        expand: Linear layer and ReLU, the hidden layer of the decoder.
        prior: Linear layer giving mean and log-variance of the prior.
        posterior: Linear layer giving the shifts of mean and log-variance from the prior.
        merge: Linear layer joining hidden features, latent and condition.
    """

    expand: Final[nn.Sequential]
    prior: Final[nn.Linear]
    posterior: Final[nn.Linear]
    merge: Final[nn.Linear]

    def __init__(self, in_dim: int, h_dim: int, z_dim: int) -> None:
        """Initialize.

        Args:
            in_dim: width of the input of the stage.
            h_dim: width of the hidden layer, the same as the matching one of the encoder.
            z_dim: width of the latent.
        """
        super().__init__()
        self.expand = nn.Sequential(nn.Linear(in_dim, h_dim), nn.ReLU())
        self.prior = nn.Linear(h_dim + z_dim, 2 * z_dim)
        self.posterior = nn.Linear(2 * h_dim, 2 * z_dim)
        self.merge = nn.Linear(h_dim + z_dim + 1, h_dim)
        return


class CounterfactualVAE(nn.Module):
    """Hierarchical VAE whose top latent has the probability of the classifier as prior mean.

    Changing the first coordinate of the top latent and the condition to a target probability
    gives a reconstruction that the classifier should assign to the target.

    Attributes:
        in_dim: Connections of each connectivity matrix.
        hidden_dims: Widths of the hidden layers, one latent stage for each.
        z_dim: Width of each latent stage.
        encoder: Hidden layers of the encoder, then the linear layer giving mean and log-variance
            of the top latent.
        stages: Latent stages of the decoder, from the top.
        head: Linear layer mapping the last stage to the connections.
    """

    in_dim: Final[int]
    hidden_dims: Final[tuple[int, ...]]
    z_dim: Final[int]
    encoder: Final[nn.ModuleList]
    stages: Final[nn.ModuleList]
    head: Final[nn.Linear]

    def __init__(self) -> None:
        """Initialize."""
        super().__init__()
        cfg = Experiment.get_config()
        cfg_model = cfg.counterfactual.model
        self.in_dim = N_FEATURES
        self.hidden_dims = cfg_model.hidden_dims
        self.z_dim = cfg_model.z_dim
        widths = [self.in_dim + 1, *self.hidden_dims]
        self.encoder = nn.ModuleList()
        for in_dim, out_dim in itertools.pairwise(widths):
            self.encoder.append(nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU()))

        self.encoder.append(nn.Linear(self.hidden_dims[-1], 2 * self.z_dim))
        self.stages = nn.ModuleList()
        in_dim = self.z_dim + 1
        for h_dim in reversed(self.hidden_dims):
            self.stages.append(LatentStage(in_dim, h_dim, self.z_dim))
            in_dim = h_dim

        self.head = nn.Linear(self.hidden_dims[0], self.in_dim)
        return

    def forward(self, inputs: Inputs, /) -> Outputs:
        """Reconstruct the connections under the condition of the inputs."""
        encoding = self.encode(inputs.features, inputs.condition)
        outputs = self.decode(encoding.z, inputs.condition, encoding.hidden)
        outputs.mu.insert(0, encoding.mu)
        outputs.log_var.insert(0, encoding.log_var)
        return outputs

    @override
    def extra_repr(self) -> str:
        return f'z_dim={self.z_dim}'

    def counterfactual(
        self, features: torch.Tensor, condition: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate counterfactual connectomes for a target probability.

        The reconstruction under the target replaces the one under the condition, so that the
        part of the connections the model does not reconstruct carries over unchanged.

        Args:
            features: connections of the subjects.
            condition: probability of the conditioning classifier for the subjects.
            target: probability the counterfactuals aim at.

        Returns:
            The counterfactuals clipped to correlations in [-1, 1], the reconstructions under
            the condition, and the ones under the target.
        """
        encoding = self.encode(features, condition)
        recon = self.decode(encoding.z, condition, encoding.hidden).recon
        z_target = encoding.z.clone()
        z_target[:, :1] = target
        recon_target = self.decode(z_target, target, encoding.hidden).recon
        counterfactuals = torch.clip(features - recon + recon_target, min=-1, max=1)
        return counterfactuals, recon, recon_target

    def decode(self, z: torch.Tensor, condition: torch.Tensor, hidden: tuple[torch.Tensor, ...] | None) -> Outputs:
        """Decode the top latent under a condition.

        Args:
            z: sample of the top latent.
            condition: probability of the conditioning classifier.
            hidden: outputs of the hidden layers of the encoder, from the deepest; ``None``
                samples the latent stages from their prior.

        Returns:
            The reconstruction, with the prior mean of the top latent and, when the hidden
            layers are given, the posterior shifts and the prior log-variances of the stages.
        """
        mus: list[torch.Tensor] = []
        log_vars: list[torch.Tensor] = []
        prior_log_vars: list[torch.Tensor] = []
        x = torch.cat([z, condition], dim=1)
        for i, stage in enumerate(cast(list[LatentStage], list(self.stages))):
            x = stage.expand(x)
            p_mu, p_log_var = stage.prior(torch.cat([x, z], dim=1)).chunk(2, 1)
            if hidden is None:
                z = self.sample(p_mu, p_log_var)
            else:
                d_mu, d_log_var = stage.posterior(torch.cat([x, hidden[i]], dim=1)).chunk(2, 1)
                z = self.sample(p_mu + d_mu, p_log_var + d_log_var)
                mus.append(d_mu)
                log_vars.append(d_log_var)
                prior_log_vars.append(p_log_var)

            x = stage.merge(torch.cat([x, z, condition], dim=1))

        prior_mu = self._get_prior_mu(condition)
        return Outputs(self.head(x), prior_mu, mu=mus, log_var=log_vars, prior_log_var=prior_log_vars)

    def encode(self, features: torch.Tensor, condition: torch.Tensor) -> Encoding:
        """Encode the connections under a condition.

        Args:
            features: connections of the subjects.
            condition: probability of the conditioning classifier.

        Returns:
            The top latent with its posterior, and the outputs of the hidden layers.
        """
        x = torch.cat([features, condition], dim=1)
        hidden: list[torch.Tensor] = []
        for layer in self.encoder[:-1]:
            x = layer(x)
            hidden.append(x)

        mu, log_var = self.encoder[-1](x).chunk(2, 1)
        return Encoding(z=self.sample(mu, log_var), mu=mu, log_var=log_var, hidden=tuple(reversed(hidden)))

    def generate(self, condition: torch.Tensor, /) -> torch.Tensor:
        """Generate connections under a condition, sampling every latent from its prior.

        Args:
            condition: probability of the conditioning classifier for each sample.

        Returns:
            The generated connections.
        """
        z = torch.randn(len(condition), self.z_dim, device=condition.device)
        z[:, :1] = condition
        return self.decode(z, condition, None).recon

    def sample(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Sample a Gaussian with the reparametrization trick, or take its mean in evaluation mode.

        Args:
            mu: the mean.
            log_var: the log-variance.

        Returns:
            The sample.
        """
        if not self.training:
            return mu

        return mu + torch.randn_like(mu) * torch.exp(0.5 * log_var)

    def _get_prior_mu(self, condition: torch.Tensor) -> torch.Tensor:
        prior_mu = torch.zeros(len(condition), self.z_dim, device=condition.device)
        prior_mu[:, :1] = condition
        return prior_mu
