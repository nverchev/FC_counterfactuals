"""Evaluate the counterfactual VAE and its counterfactuals on the evaluation partition.

Usage::

    uv run evaluate_counterfactuals.py final=true
    uv run evaluate_counterfactuals.py --config-name mlp final=true
"""

from drytorch import Test
from drytorch.core import log_events

from src.config import ExperimentConfig, Experiment, hydra_main
from src.config.options import CounterfactualTargets
from src.trackers import get_trackers
from src.config.environment import DEBUG_MODE
from src.data import get_splits
from src.lib import get_loaders
from src.data.conditions import condition_split
from src.utils.counterfactuals import evaluate_counterfactuals
from src.lib.models import get_counterfactual_model, load_classifier_probability, load_trained_state
from src.lib.objectives import get_vae_loss


def evaluate() -> None:
    """Evaluate the reconstructions and the counterfactuals of the trained VAE.

    Side Effects:
        Publishes the metrics of the counterfactuals of each target, as the source
        ``Counterfactual_<target>``, and of the denoised ones, as ``Denoised_counterfactual_<target>``.
    """
    cfg = Experiment.get_config()
    probability = load_classifier_probability()
    _, eval_split = get_splits()
    eval_split = condition_split(probability, eval_split)
    model = get_counterfactual_model()
    load_trained_state(model, epoch=cfg.user.load or -1)

    _, eval_loader = get_loaders(eval_split, eval_split, batch_size=cfg.counterfactual.train.batch_size)
    Test(model, loader=eval_loader, metric=get_vae_loss(), name='Reconstruction')()
    module = model.module
    for target in CounterfactualTargets:
        noisy = evaluate_counterfactuals(module, probability, eval_split, denoise=False, target=target)
        log_events.MetricEvent(
            model_name=model.name, source_name=f'Counterfactual_{target}', epoch=model.epoch, metrics=noisy.metrics
        )
        denoised = evaluate_counterfactuals(module, probability, eval_split, denoise=True, target=target)
        log_events.MetricEvent(
            model_name=model.name,
            source_name=f'Denoised_counterfactual_{target}',
            epoch=model.epoch,
            metrics=denoised.metrics,
        )

    return


@hydra_main
def main(cfg: ExperimentConfig) -> None:
    """Set up the experiment and evaluate the counterfactuals."""
    trackers = get_trackers(cfg)
    exp = Experiment(cfg, name=cfg.name, par_dir=cfg.par_dir, tags=cfg.tags)
    for tracker in trackers:
        exp.trackers.subscribe(tracker)

    with exp.create_run(resume=True, run_id=cfg.user.run_id, record=not DEBUG_MODE):
        evaluate()

    return


if __name__ == '__main__':
    main()
