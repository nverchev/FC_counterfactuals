"""Setup of the trackers of the experiment."""

import os
import warnings

from hydra.core import utils as hydra_utils

import drytorch

from drytorch.core import tracking
from drytorch.trackers import logging

from src.config.environment import DEBUG_MODE
from src.config.specs import ExperimentConfig
from src.trackers.logging import ClassificationLogger


__all__ = [
    'get_trackers',
    'set_tuning_logging',
]


def get_trackers(cfg: ExperimentConfig, debug_mode: bool = DEBUG_MODE) -> list[tracking.Tracker]:
    """Instantiate the experiment trackers requested in *cfg*.

    Trackers are only activated outside debug mode.  In debug mode an empty
    list is returned so that no files or remote metrics are written.

    Args:
        cfg: Root configuration object.
        debug_mode: When ``True``, skip all tracker initialisation. Defaults to
            whether a debugger runs the process.

    Returns:
        A list of initialised :class:`drytorch.core.tracking.Tracker` objects
        ready to be passed to :meth:`drytorch.Experiment.trackers.subscribe`.
    """
    cfg_trackers = cfg.user.trackers
    cfg_hydra = cfg.user.hydra
    hydra_utils.configure_log(cfg_hydra.job_logging)
    drytorch.init_trackers(mode='hydra')

    # optionally replace the built-in logger with a project-specific one
    tracking.DEFAULT_TRACKERS.pop(logging.BuiltinLogger.__name__, None)
    drytorch.extend_default_trackers([ClassificationLogger()])
    optional_tracker_list: list[tracking.Tracker] = []
    if debug_mode:
        return optional_tracker_list

    if cfg_trackers.wandb:
        try:
            import wandb

            from src.trackers.wandb import ClassificationWandb

        except ImportError as e:
            warnings.warn(f'Wandb not installed, skipping. {e}', stacklevel=2)

        else:
            optional_tracker_list.append(ClassificationWandb(settings=wandb.Settings(project=cfg.project)))

    if cfg_trackers.hydra:
        from drytorch.trackers.hydra import HydraLink

        optional_tracker_list.append(HydraLink(hydra_dir=cfg.user.hydra.output_dir))

    if cfg_trackers.csv:
        from drytorch.trackers.csv import CSVDumper

        optional_tracker_list.append(CSVDumper())

    if cfg_trackers.tensorboard:
        try:
            from src.trackers.tensorboard import ClassificationTensorBoard

        except ImportError as e:
            warnings.warn(f'TensorBoard not installed, skipping. {e}', stacklevel=2)

        else:
            optional_tracker_list.append(ClassificationTensorBoard())

    if cfg_trackers.sqlalchemy:
        try:
            import sqlalchemy

            from drytorch.trackers.sqlalchemy import SQLConnection

        except ImportError as e:
            warnings.warn(f'SQLAlchemy not installed, skipping. {e}', stacklevel=2)

        else:
            engine_path = cfg.par_dir / 'metrics.db'
            cfg.par_dir.mkdir(parents=True, exist_ok=True)
            engine = sqlalchemy.create_engine(f'sqlite:///{engine_path}')
            optional_tracker_list.append(SQLConnection(engine=engine))

    return optional_tracker_list


def set_tuning_logging() -> None:
    """Configure minimal console logging for parallel Optuna workers.

    When GNU parallel runs the workers, it sets ``PARALLEL_SEQ`` in each of them, and the
    output is restricted to test metrics to prevent interleaved logs across workers.
    """
    if 'PARALLEL_SEQ' in os.environ:
        drytorch.remove_all_default_trackers()
        drytorch.extend_default_trackers([logging.BuiltinLogger()])
        logging.set_verbosity(logging.INFO_LEVELS.test)
    else:
        drytorch.init_trackers(mode='minimal')

    return
