"""Loading of a trained experiment for the analyses of the reconstruction error."""

import dataclasses
import pathlib
from collections.abc import Callable, Sequence
from typing import cast

import numpy as np
import torch

from src.config import ExperimentConfig, Experiment, get_experiment_config
from src.data import get_splits
from src.module import CounterfactualVAE
from src.data.conditions import condition_split
from src.lib.models import load_classifier_probability

__all__ = [
    'AnalysisSetup',
    'load_setup',
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


def load_setup(classifier: str, overrides: Sequence[str] = ('final=true',)) -> AnalysisSetup:
    """Load the trained counterfactual VAE conditioned on a classifier.

    Args:
        classifier: the conditioning classifier, which names the entry configuration: ``svc``, ``mlp`` or ``ae``.
        overrides: the Hydra overrides the models were trained with.

    Returns:
        The models and the partitions of the experiment.

    Raises:
        RuntimeError: if the experiment holds no trained models.
    """
    cfg = get_experiment_config(classifier, overrides=list(overrides))
    exp = Experiment(cfg, name=cfg.name, par_dir=cfg.par_dir)
    with exp.create_run(resume=True, record=False):
        probability = load_classifier_probability()
        train_split, test_split = get_splits()
        splits = {'train': condition_split(probability, train_split), 'test': condition_split(probability, test_split)}

        # drytorch numbers a repeated model name in a process, so the weights are read directly
        module = CounterfactualVAE()
        module.load_state_dict(torch.load(_get_last_state(cfg), map_location='cpu', weights_only=True))

    partitions = {
        name: (split.features.numpy(), split.labels.squeeze(1).long().numpy(), split.conditions.squeeze(1).numpy())
        for name, split in splits.items()
    }
    return AnalysisSetup(module=module.eval(), probability=probability, partitions=partitions)


def _get_last_state(cfg: ExperimentConfig) -> pathlib.Path:
    # the checkpoints are in <run name>/<date>/<time>/<model name>/epoch_<epoch>
    model_dirs = sorted((cfg.par_dir / 'checkpoints' / cfg.name).glob(f'*/*/{cfg.counterfactual.model.name}'))
    epoch_dirs = list(model_dirs[-1].glob('epoch_*')) if model_dirs else []
    if not epoch_dirs:
        msg = f"No trained counterfactual VAE in the experiment '{cfg.name}': train it first."
        raise RuntimeError(msg)

    return max(epoch_dirs, key=lambda path: int(path.name.removeprefix('epoch_'))) / 'model_state.pt'
