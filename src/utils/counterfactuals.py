"""Generation and evaluation of counterfactual connectomes."""

import dataclasses
from collections.abc import Callable

import torch
import torch.nn.functional as F

from src.config.options import CounterfactualTargets
from src.data.split import DataSplit
from src.module import CounterfactualVAE

__all__ = [
    'CounterfactualResults',
    'evaluate_counterfactuals',
    'get_targets',
]


@dataclasses.dataclass(frozen=True)
class CounterfactualResults:
    """Counterfactuals of a partition and their metrics.

    Attributes:
        counterfactuals: Counterfactual connections.
        targets: Probabilities the counterfactuals aim at.
        probabilities: Probabilities the classifier assigns to the counterfactuals.
        metrics: Distances from the originals and effect on the classifier.
    """

    counterfactuals: torch.Tensor
    targets: torch.Tensor
    probabilities: torch.Tensor
    metrics: dict[str, float]


@torch.inference_mode()
def evaluate_counterfactuals(
    module: CounterfactualVAE,
    probability: Callable[[torch.Tensor], torch.Tensor],
    split: DataSplit,
    *,
    denoise: bool,
    target: CounterfactualTargets,
) -> CounterfactualResults:
    """Generate the counterfactuals of a partition and measure them.

    The counterfactuals are the reconstructions under the target. Denoising removes their
    reconstruction error by adding the residual, the part of the original that the
    reconstruction under the condition misses.

    The metrics are:

    - ``MSE`` and ``MAE``: distance of the counterfactuals from the originals;
    - ``RMS_from_half``: root mean square distance of the new probabilities from 0.5;
    - ``CFD``: root mean square distance of the new probabilities from the conditions;
    - ``CFD_all_pairs``: the same distance over every pair of subjects, as version 1 computed
      it by mistake, for the comparison with the published results;
    - ``Success``: fraction of the predictions that the counterfactuals flip, which is the DBS
      when the target is 0.5.

    Args:
        module: the trained counterfactual VAE.
        probability: the classifier, from connections to probabilities.
        split: the partition, conditioned on the probabilities of the classifier.
        denoise: whether to remove the reconstruction error from the counterfactuals.
        target: probability the counterfactuals aim at.

    Returns:
        The counterfactuals with their metrics.
    """
    module.eval()
    device = next(module.parameters()).device
    features = split.features.to(device)
    conditions = split.conditions.to(device)
    targets = get_targets(conditions, target)
    denoised, _, recon_target = module.counterfactual(features, conditions, targets)
    counterfactuals = denoised if denoise else torch.clip(recon_target, min=-1, max=1)
    new_probabilities = probability(counterfactuals)
    flipped = (conditions >= 0.5) != (new_probabilities >= 0.5)
    metrics = {
        'MSE': F.mse_loss(counterfactuals, features).item(),
        'MAE': F.l1_loss(counterfactuals, features).item(),
        'RMS_from_half': (new_probabilities - 0.5).pow(2).mean().sqrt().item(),
        'CFD': (new_probabilities - conditions).pow(2).mean().sqrt().item(),
        'CFD_all_pairs': (new_probabilities.view(1, -1) - conditions).pow(2).mean().sqrt().item(),
        'Success': flipped.float().mean().item(),
    }
    return CounterfactualResults(counterfactuals.cpu(), targets.cpu(), new_probabilities.cpu(), metrics)


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
