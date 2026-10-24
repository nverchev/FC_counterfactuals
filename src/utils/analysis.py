"""Trained counterfactual VAE and classifier, with NumPy inputs, for the analyses of the reconstruction error."""

import dataclasses
from collections.abc import Callable
from typing import cast

import numpy as np
import torch

from src.module import CounterfactualVAE

__all__ = [
    'AnalysisSetup',
]


@dataclasses.dataclass(frozen=True)
class AnalysisSetup:
    """Trained counterfactual VAE, its classifier and the partitions, as NumPy arrays.

    Attributes:
        module: The counterfactual VAE, in evaluation mode.
        probability: The classifier, from connections to probabilities.
        partitions: Connections, labels and conditions of the training (``train``) and the
            evaluation (``test``) partitions.
    """

    module: CounterfactualVAE
    probability: Callable[[torch.Tensor], torch.Tensor]
    partitions: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]

    def classify(self, features: np.ndarray, /) -> np.ndarray:
        """Get the probabilities of the classifier.

        Args:
            features: connections ``(N, n_features)``.

        Returns:
            The probabilities ``(N,)``.
        """
        return self.probability(torch.from_numpy(features).float()).squeeze(1).numpy()

    @torch.inference_mode()
    def counterfactuals(
        self, features: np.ndarray, conditions: np.ndarray, targets: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get the counterfactuals and the reconstructions under the conditions and the targets.

        Args:
            features: connections ``(N, n_features)``.
            conditions: probabilities of the classifier ``(N,)``.
            targets: probabilities the counterfactuals aim at ``(N,)``.

        Returns:
            The counterfactuals, the reconstructions under the conditions, and the ones under
            the targets.
        """
        device = next(self.module.parameters()).device
        tensors = [torch.from_numpy(array).float().to(device) for array in (features, conditions, targets)]
        x, condition, target = tensors[0], tensors[1].view(-1, 1), tensors[2].view(-1, 1)
        outputs = self.module.counterfactual(x, condition, target)
        return cast(tuple[np.ndarray, np.ndarray, np.ndarray], tuple(t.cpu().numpy() for t in outputs))

    @torch.inference_mode()
    def reconstruct(self, features: np.ndarray, conditions: np.ndarray) -> np.ndarray:
        """Reconstruct the connections under their conditions, without sampling.

        Args:
            features: connections ``(N, n_features)``.
            conditions: probabilities of the classifier ``(N,)``.

        Returns:
            The reconstructions.
        """
        device = next(self.module.parameters()).device
        x = torch.from_numpy(features).float().to(device)
        condition = torch.from_numpy(conditions).float().view(-1, 1).to(device)
        encoding = self.module.encode(x, condition)
        return self.module.decode(encoding.z, condition, encoding.hidden).recon.cpu().numpy()
