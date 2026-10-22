"""Train the classifier whose probabilities condition the counterfactual VAE.

Usage::

    uv run train_classifier.py                       # support vector classifier
    uv run train_classifier.py --config-name mlp
    uv run train_classifier.py --config-name ae  # after train_autoencoder.py
    uv run train_classifier.py final=true
"""

from drytorch import Trainer

from src.config import ExperimentConfig, Experiment, hydra_main
from src.trackers import get_trackers
from src.config.environment import DEBUG_MODE
from src.config.options import Classifiers
from src.data import DataSplit, get_splits
from src.lib import get_learning_schema, get_loaders
from src.lib.hooks import register_checkpointing, register_early_stopping
from src.lib.models import get_mlp_model, load_pretrained_encoder
from src.lib.objectives import get_classifier_loss
from src.utils.svc import fit_and_save_svc


def train_mlp(train_split: DataSplit, eval_split: DataSplit) -> None:
    """Train the MLP classifier, from the encoder of the autoencoder for ``ae``.

    Args:
        train_split: split of the training partition.
        eval_split: split of the validation partition, used only without ``final``.

    Side Effects:
        Saves the checkpoint of the classifier in the folder of the experiment.
    """
    cfg = Experiment.get_config()
    cfg_classification = cfg.classification
    cfg_train = cfg_classification.train
    model = get_mlp_model()
    if cfg_classification.model.name == Classifiers.AE and not cfg.user.load:
        load_pretrained_encoder(model.module)

    train_loader, eval_loader = get_loaders(
        train_split, eval_split, batch_size=cfg_train.batch_size, n_workers=cfg.user.n_workers
    )
    builder = Trainer.builder(
        model,
        learning_schema=get_learning_schema(cfg_train),
        loader=train_loader,
        loss=get_classifier_loss(),
    )
    if not cfg.final:
        builder.add_validation(eval_loader)

    if not cfg.final and cfg_train.early_stopping.active:
        register_early_stopping(
            builder, window=cfg_train.early_stopping.window, patience=cfg_train.early_stopping.patience
        )

    register_checkpointing(builder, checkpoint_every=cfg.user.checkpoint_every)
    trainer = builder.build()
    if cfg.user.load:
        trainer.load_checkpoint(cfg.user.load)

    trainer.train_until(cfg_train.n_epochs)
    trainer.save_checkpoint()
    return


def train_classifier() -> None:
    """Train or fit the configured classifier and save it.

    Side Effects:
        Saves the classifier in the folder of the experiment.
    """
    cfg = Experiment.get_config()
    train_split, eval_split = get_splits()
    if cfg.classification.model.name == Classifiers.SVC:
        fit_and_save_svc(train_split)
    else:
        train_mlp(train_split, eval_split)

    return


@hydra_main
def main(cfg: ExperimentConfig) -> None:
    """Set up the experiment and train the classifier."""
    trackers = get_trackers(cfg)
    exp = Experiment(cfg, name=cfg.name, par_dir=cfg.par_dir, tags=cfg.tags)
    for tracker in trackers:
        exp.trackers.subscribe(tracker)

    classifier = cfg.classification.model.name
    resume = cfg.user.load != 0 or cfg.user.run_id is not None or classifier == Classifiers.AE
    with exp.create_run(resume=resume, run_id=cfg.user.run_id, record=not DEBUG_MODE):
        train_classifier()

    return


if __name__ == '__main__':
    main()
