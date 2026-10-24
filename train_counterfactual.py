"""Train the counterfactual VAE conditioned on the probabilities of the trained classifier.

Usage::

    uv run train_counterfactual.py                               # after the svc classifier
    uv run train_counterfactual.py --config-name mlp
    uv run train_counterfactual.py final=true
"""

from drytorch import Trainer

from src.config import ExperimentConfig, Experiment, hydra_main
from src.trackers import get_trackers
from src.config.environment import DEBUG_MODE
from src.data import get_splits
from src.lib import get_learning_schema, get_loaders
from src.data.derived_datasets import ConditionedSplit, load_classifier
from src.lib.hooks import register_checkpointing
from src.lib.models import get_counterfactual_model
from src.lib.objectives import get_vae_loss


def train_counterfactual() -> None:
    """Train the counterfactual VAE and save it."""
    cfg = Experiment.get_config()
    cfg_train = cfg.counterfactual.train
    train_split, eval_split = get_splits()
    classifier = load_classifier()
    train_split = ConditionedSplit(train_split, classifier)
    eval_split = ConditionedSplit(eval_split, classifier)
    model = get_counterfactual_model()
    train_loader, eval_loader = get_loaders(
        train_split, eval_split, batch_size=cfg_train.batch_size, n_workers=cfg.user.n_workers
    )
    builder = Trainer.builder(
        model,
        learning_schema=get_learning_schema(cfg_train),
        loader=train_loader,
        loss=get_vae_loss(),
    )
    if not cfg.final:
        builder.add_validation(eval_loader)

    register_checkpointing(builder, checkpoint_every=cfg.user.checkpoint_every)
    trainer = builder.build()
    if cfg.user.load:
        trainer.load_checkpoint(cfg.user.load)

    trainer.train_until(cfg_train.n_epochs)
    trainer.save_checkpoint()
    return


@hydra_main
def main(cfg: ExperimentConfig) -> None:
    """Set up the experiment and train the counterfactual VAE."""
    trackers = get_trackers(cfg)
    exp = Experiment(cfg, name=cfg.name, par_dir=cfg.par_dir, tags=cfg.tags)
    for tracker in trackers:
        exp.trackers.subscribe(tracker)

    with exp.create_run(resume=True, run_id=cfg.user.run_id, record=not DEBUG_MODE):
        train_counterfactual()

    return


if __name__ == '__main__':
    main()
