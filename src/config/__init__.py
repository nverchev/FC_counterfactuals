"""Package for configuring the experiment."""

from src.config.specs import ExperimentConfig
from src.config.hydra import hydra_main, get_experiment_config
from src.config.experiment import Experiment
from src.config.environment import PROJECT_NAME, VERSION, ConfigPath

__all__ = [
    'PROJECT_NAME',
    'VERSION',
    'ConfigPath',
    'Experiment',
    'ExperimentConfig',
    'get_experiment_config',
    'hydra_main',
]
