"""Train the autoencoder that pretrains the encoder of the ``ae`` classifier.

Usage::

    uv run train_autoencoder.py
    uv run train_autoencoder.py final=true
"""

from drytorch import Trainer

from src.config import ExperimentConfig, Experiment, hydra_main
from src.trackers import get_trackers
from src.config.environment import DEBUG_MODE
from src.data import get_splits
from src.lib import get_learning_schema, get_loaders
from src.lib.hooks import register_checkpointing
from src.lib.models import get_autoencoder_model
from src.lib.objectives import get_autoencoder_loss


def train_autoencoder() -> None:
    """Train the autoencoder and save it."""
    cfg = Experiment.get_config()
    cfg_train = cfg.autoencoding.train
    model = get_autoencoder_model()
    train_split, eval_split = get_splits()
    train_loader, eval_loader = get_loaders(
        train_split, eval_split, batch_size=cfg_train.batch_size, n_workers=cfg.user.n_workers
    )
    builder = Trainer.builder(
        model,
        learning_schema=get_learning_schema(cfg_train),
        loader=train_loader,
        loss=get_autoencoder_loss(),
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
    """Set up the experiment and train the autoencoder."""
    trackers = get_trackers(cfg)
    exp = Experiment(cfg, name=cfg.name, par_dir=cfg.par_dir, tags=cfg.tags)
    for tracker in trackers:
        exp.trackers.subscribe(tracker)

    resume = cfg.user.load != 0 or cfg.user.run_id is not None
    with exp.create_run(resume=resume, run_id=cfg.user.run_id, record=not DEBUG_MODE):
        train_autoencoder()

    return


if __name__ == '__main__':
    main()
