"""Publish the reconstructions and the counterfactuals of some subjects of the evaluation partition.

Usage::

    uv run visualize_counterfactuals.py final=true
    uv run visualize_counterfactuals.py final=true "user.plot.sample_indices=[0,5,7]"
"""

import logging

from drytorch.lib import loading

from src.config import ExperimentConfig, Experiment, hydra_main
from src.config.options import CounterfactualTargets
from src.data import get_splits
from src.data.abide import AbideDataBuilder
from src.trackers import get_trackers
from src.data.derived_datasets import ConditionedSplit, CounterfactualSplit, ReconstructedSplit, load_classifier
from src.lib.models import get_counterfactual_model, load_trained_state
from src.utils.plotting import LABEL_NAMES, describe_probability, get_region_coordinates, publish_connectome


def visualize() -> None:
    """Publish the figures of the configured subjects.

    Side Effects:
        Publishes a reconstruction and a counterfactual figure for each subject.
    """
    cfg = Experiment.get_config()
    classifier = load_classifier()
    _, eval_split = get_splits()
    eval_split = ConditionedSplit(eval_split, classifier)
    model = get_counterfactual_model()
    load_trained_state(model, epoch=cfg.user.load or -1)

    if cfg.user.plot.sample_indices:
        indices, _ = loading.take_from_dataset(eval_split, indices=cfg.user.plot.sample_indices)
    else:
        indices, _ = loading.take_from_dataset(
            eval_split, n_samples=cfg.counterfactual.train.batch_size, preserve_order=False
        )

    derived_splits: dict[str, ReconstructedSplit | CounterfactualSplit] = {}
    derived_splits['reconstruction'] = ReconstructedSplit(eval_split, model.module, classifier)
    for target in CounterfactualTargets:
        counterfactuals = CounterfactualSplit(eval_split, model.module, classifier, denoise=True, target=target)
        derived_splits[f'target_{target}'] = counterfactuals

    coordinates = get_region_coordinates(AbideDataBuilder().atlas_path)
    partition = 'test' if cfg.final else 'val'
    for index in indices:
        subject_id = eval_split.subject_ids[index]
        folder = f'{partition}/subject_{subject_id}'
        label = LABEL_NAMES[int(eval_split.labels[index].item())]
        caption = f'{label}, {describe_probability(eval_split.probabilities[index].item())}'
        features = eval_split.features[index]
        publish_connectome(features, coordinates=coordinates, title='original', caption=caption, folder=folder)
        for title, split in derived_splits.items():
            caption = describe_probability(split.probabilities[index].item())
            publish_connectome(
                split.features[index], coordinates=coordinates, title=title, caption=caption, folder=folder
            )

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
