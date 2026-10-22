"""Publish the reconstructions and the counterfactuals of some subjects of the evaluation partition.

Usage::

    uv run visualize_counterfactuals.py final=true
    uv run visualize_counterfactuals.py final=true "user.plot.sample_indices=[0,5,7]"
"""

import logging

import torch

from drytorch.lib import loading

from src.config import ExperimentConfig, Experiment, hydra_main
from src.config.options import CounterfactualTargets
from src.data import get_splits
from src.trackers import get_trackers
from src.data.conditions import condition_split
from src.utils.counterfactuals import get_targets
from src.lib.models import get_counterfactual_model, load_classifier_probability, load_trained_state
from src.utils.plotting import LABEL_NAMES, describe_probability, publish_connectome


@torch.inference_mode()
def visualize() -> None:
    """Publish the figures of the configured subjects.

    Side Effects:
        Publishes a reconstruction and a counterfactual figure for each subject.
    """
    cfg = Experiment.get_config()
    probability = load_classifier_probability()
    _, eval_split = get_splits()
    eval_split = condition_split(probability, eval_split)
    model = get_counterfactual_model()
    load_trained_state(model, epoch=cfg.user.load or -1)

    module = model.module.eval()
    if cfg.user.plot.sample_indices:
        indices, (inputs, targets) = loading.take_from_dataset(
            eval_split, indices=cfg.user.plot.sample_indices, device=model.device
        )
    else:
        indices, (inputs, targets) = loading.take_from_dataset(
            eval_split, n_samples=cfg.counterfactual.train.batch_size, preserve_order=False, device=model.device
        )

    # in evaluation mode the latent is the posterior mean, so every target shares the reconstruction
    counterfactuals = {}
    for target in CounterfactualTargets:
        goals = get_targets(inputs.condition, target)
        counterfactuals[target], recon, _ = module.counterfactual(inputs.features, inputs.condition, goals)

    original_probabilities = probability(inputs.features).squeeze(1).tolist()
    recon_probabilities = probability(recon).squeeze(1).tolist()
    counterfactual_probabilities = {
        target: probability(counterfactual).squeeze(1).tolist() for target, counterfactual in counterfactuals.items()
    }
    partition = 'test' if cfg.final else 'val'
    for i, index in enumerate(indices):
        subject_id = eval_split.subject_ids[index]
        folder = f'{partition}/subject_{subject_id}'
        label = LABEL_NAMES[int(targets.label[i].item())]
        caption = f'{label}, {describe_probability(original_probabilities[i])}'
        publish_connectome(inputs.features[i], title='original', caption=caption, folder=folder)
        caption = describe_probability(recon_probabilities[i])
        publish_connectome(recon[i], title='reconstruction', caption=caption, folder=folder)
        for target, counterfactual in counterfactuals.items():
            caption = describe_probability(counterfactual_probabilities[target][i])
            publish_connectome(counterfactual[i], title=f'target_{target}', caption=caption, folder=folder)

        logging.info('Published the figures of %s.', subject_id)

    return


@hydra_main
def main(cfg: ExperimentConfig) -> None:
    """Set up the experiment and publish the figures."""
    trackers = get_trackers(cfg)
    exp = Experiment(cfg, name=cfg.name, par_dir=cfg.par_dir, tags=cfg.tags)
    for tracker in trackers:
        exp.trackers.subscribe(tracker)

    with exp.create_run(resume=True, run_id=cfg.user.run_id):
        visualize()

    return


if __name__ == '__main__':
    main()
