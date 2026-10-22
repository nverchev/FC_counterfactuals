"""Tests for the autoencoder and the counterfactual VAE."""

import pytest
import torch

from src.data.structures import Inputs
from src.module.autoencoders import Autoencoder, CounterfactualVAE

from src.data.abide import N_FEATURES


class TestAutoencoder:
    """Test the reconstruction of the autoencoder."""

    def test_reconstructs_the_connections(self) -> None:
        """The reconstruction has the shape of the connections."""
        inputs = Inputs(features=torch.randn(3, N_FEATURES), condition=torch.zeros(3, 1))

        recon = Autoencoder()(inputs)

        assert recon.shape == (3, N_FEATURES)


class TestCounterfactualVAE:
    """Test the reconstructions and the counterfactuals of the VAE."""

    @pytest.fixture
    def module(self) -> CounterfactualVAE:
        """Provide the VAE in evaluation mode."""
        return CounterfactualVAE().eval()

    @pytest.fixture
    def inputs(self) -> Inputs:
        """Provide connections in [-1, 1] with their conditions."""
        return Inputs(features=torch.rand(5, N_FEATURES) * 2 - 1, condition=torch.rand(5, 1))

    def test_outputs_have_one_posterior_for_each_stage(self, module: CounterfactualVAE, inputs: Inputs) -> None:
        """The top stage and one stage for each hidden layer have a posterior."""
        outputs = module(inputs)

        assert outputs.recon.shape == (5, N_FEATURES)
        assert len(outputs.mu) == len(outputs.log_var) == 1 + len(module.hidden_dims)
        assert len(outputs.prior_log_var) == len(module.hidden_dims)

    def test_prior_mean_of_the_top_stage_is_the_condition(self, module: CounterfactualVAE, inputs: Inputs) -> None:
        """The condition sets the first coordinate of the prior mean, the others are zero."""
        outputs = module(inputs)

        torch.testing.assert_close(outputs.prior_mu[:, :1], inputs.condition)
        torch.testing.assert_close(outputs.prior_mu[:, 1:], torch.zeros(5, module.z_dim - 1))

    def test_evaluation_is_deterministic(self, module: CounterfactualVAE, inputs: Inputs) -> None:
        """Without sampling, the same inputs give the same reconstruction."""
        first, second = module(inputs).recon, module(inputs).recon

        torch.testing.assert_close(first, second)

    def test_counterfactuals_are_correlations(self, module: CounterfactualVAE, inputs: Inputs) -> None:
        """The counterfactuals are clipped to [-1, 1]."""
        counterfactuals, recon, recon_target = module.counterfactual(
            inputs.features, inputs.condition, 1 - inputs.condition
        )

        assert counterfactuals.shape == recon.shape == recon_target.shape == (5, N_FEATURES)
        assert counterfactuals.abs().max() <= 1

    def test_generation_samples_the_priors(self, module: CounterfactualVAE) -> None:
        """Generated connections have the shape of the data."""
        generated = module.generate(torch.tensor([[0.0], [1.0]]))

        assert generated.shape == (2, N_FEATURES)

    def test_sample_in_evaluation_is_the_mean(self, module: CounterfactualVAE) -> None:
        """In evaluation mode, the sample is the mean."""
        mu = torch.randn(3, 2)

        result = module.sample(mu, torch.zeros(3, 2))

        torch.testing.assert_close(result, mu)

    def test_sample_in_training_spreads_with_the_variance(self, module: CounterfactualVAE) -> None:
        """In training mode, the samples spread with the standard deviation."""
        module.train()
        mu = torch.zeros(10_000, 1)

        result = module.sample(mu, torch.full((10_000, 1), 2 * torch.log(torch.tensor(3.0))))

        assert abs(result.std().item() - 3.0) < 0.1
