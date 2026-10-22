"""Tests for the generation and the evaluation of counterfactuals."""

import torch

from src.config.options import CounterfactualTargets
from src.data.split import DataSplit
from src.module import CounterfactualVAE
from src.utils.counterfactuals import evaluate_counterfactuals, get_targets

from src.data.abide import N_FEATURES


def test_get_targets() -> None:
    """The opposite target is the class the classifier does not predict; half is 0.5 for all."""
    conditions = torch.tensor([[0.2], [0.7], [0.5]])

    opposite = get_targets(conditions, CounterfactualTargets.OPPOSITE)
    half = get_targets(conditions, CounterfactualTargets.HALF)

    torch.testing.assert_close(opposite, torch.tensor([[1.0], [0.0], [0.0]]))
    torch.testing.assert_close(half, torch.full((3, 1), 0.5))


def test_evaluate_counterfactuals() -> None:
    """A classifier that always answers 0.9 flips exactly the subjects conditioned below 0.5."""
    conditions = torch.tensor([[0.2], [0.8], [0.3], [0.6]])
    split = DataSplit(torch.rand(4, N_FEATURES) * 2 - 1, torch.zeros(4, 1), conditions=conditions)

    results = evaluate_counterfactuals(
        CounterfactualVAE(),
        lambda features: torch.full((len(features), 1), 0.9),
        split,
        denoise=True,
        target=CounterfactualTargets.OPPOSITE,
    )

    assert results.metrics['Success'] == 0.5
    assert abs(results.metrics['RMS_from_half'] - 0.4) < 1e-6
    assert results.counterfactuals.shape == (4, N_FEATURES)


def test_evaluate_counterfactuals_denoising() -> None:
    """Without denoising, the counterfactuals are the reconstructions under the target; denoising adds the residual."""
    conditions = torch.tensor([[0.2], [0.8]])
    split = DataSplit(torch.rand(2, N_FEATURES) * 2 - 1, torch.zeros(2, 1), conditions=conditions)
    module = CounterfactualVAE().eval()
    targets = get_targets(conditions, CounterfactualTargets.HALF)
    expected_denoised, _, recon_target = module.counterfactual(split.features, conditions, targets)

    noisy = evaluate_counterfactuals(
        module,
        lambda features: torch.full((len(features), 1), 0.9),
        split,
        denoise=False,
        target=CounterfactualTargets.HALF,
    )
    denoised = evaluate_counterfactuals(
        module,
        lambda features: torch.full((len(features), 1), 0.9),
        split,
        denoise=True,
        target=CounterfactualTargets.HALF,
    )

    torch.testing.assert_close(noisy.counterfactuals, torch.clip(recon_target, min=-1, max=1))
    torch.testing.assert_close(denoised.counterfactuals, expected_denoised)
