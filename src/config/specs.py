"""Specification for the configuration files.

Root config hierarchy::

    ExperimentConfig
    ├── data: DataConfig
    │   └── dataset: DatasetConfig
    ├── classification: ClassificationConfig
    │   ├── svc: SVCConfig
    │   ├── model: ClassifierArchitectureConfig
    │   └── train: TrainingConfig
    ├── autoencoding: AutoencodingConfig
    │   └── train: TrainingConfig
    ├── counterfactual: CounterfactualConfig
    │   ├── model: VAEArchitectureConfig
    │   ├── objective: ObjectiveConfig
    │   └── train: TrainingConfig
    └── user: UserSettings
        ├── trackers: TrackerList
        ├── path: PathSpecs          ← populated from .env
        └── hydra: HydraSettings     ← populated at runtime

Each phase trains one model: the classifiers, the autoencoder that pretrains the
encoder of the ``ae`` classifier, and the counterfactual VAE conditioned on a classifier.
"""

import dataclasses
import pathlib
from functools import cached_property

from typing import Any, Annotated, Self, Literal

import torch

from pydantic import BeforeValidator, Field, PlainValidator, model_validator
from pydantic.dataclasses import dataclass

from src.config.environment import EnvSettings, PROJECT_NAME, VERSION, get_current_branch
from src.config.torch import get_device, get_optim_cls, set_seed
from src.config.options import Classifiers, ClipCriterion, Datasets, GradOp, Schedulers


Device = Annotated[torch.device | Literal['auto'], PlainValidator(get_device)]
OptimizerClass = Annotated[type[torch.optim.Optimizer], BeforeValidator(get_optim_cls)]
Fraction = Annotated[float, Field(gt=0, lt=1)]
PositiveInt = Annotated[int, Field(ge=0)]
StrictlyPositiveInt = Annotated[int, Field(gt=0)]
PositiveFloat = Annotated[float, Field(ge=0)]


# ---------------------------------------------------------------------------
# Data configs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetConfig:
    """Dataset location and partitions.

    Attributes:
        name: Name of the dataset.
        folder: Folder of the dataset inside the data directory.
        test_fraction: Fraction of the subjects held out for the test partition.
        val_fraction: Fraction of the remaining subjects held out for the validation partition.
        split_seed: Seed of the random partitioning.
    """

    name: Datasets
    folder: str
    test_fraction: Fraction
    val_fraction: Fraction
    split_seed: int


@dataclass(frozen=True)
class DataConfig:
    """Full data configuration.

    Attributes:
        dataset: Dataset location and partitions.
    """

    dataset: DatasetConfig


# ---------------------------------------------------------------------------
# Training configs — generic
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SchedulerConfig:
    """Specification for the learning-rate scheduler.

    Attributes:
        function: Which scheduler to use (see :class:`~src.config.options.Schedulers`).
        restart_interval: Epochs between cosine restarts (0 = no restart).
        restart_fraction: Fraction of base LR applied at each restart.
        warmup_steps: Linear warm-up epochs at the start of training.
        settings: Additional keyword arguments forwarded to the scheduler
            constructor (e.g. ``decay_steps``, ``min_decay``).
    """

    function: Schedulers
    restart_interval: PositiveInt
    restart_fraction: PositiveFloat
    warmup_steps: PositiveInt
    settings: dict[str, Any]


@dataclass(frozen=True)
class LearningConfig:
    """Specification for the optimiser and learning schedule.

    Attributes:
        optimizer: PyTorch optimizer, written by its name, e.g. ``AdamW``.
        lr: Global learning rate (or per-parameter group base LR).
        scheduler: Learning-rate scheduler configuration.
        grad_op: Optional gradient pre-processing operation applied before
            each optimiser step (see :class:`~src.config.options.GradOp`).
        clip_criterion: Criterion for histogram-based clippers.
        opt_settings: Extra keyword arguments forwarded to the optimiser
            constructor (e.g. ``weight_decay``).
    """

    optimizer: OptimizerClass
    lr: PositiveFloat
    scheduler: SchedulerConfig
    grad_op: GradOp | None
    clip_criterion: ClipCriterion | None
    opt_settings: dict[str, Any]

    @model_validator(mode='after')
    def _check_clip_criterion(self) -> Self:
        if self.clip_criterion is None and self.grad_op in (GradOp.HIST_CLIPPER, GradOp.PARAM_HIST_CLIPPER):
            msg = 'Must specify clip_criterion when using HistClipper or ParamHistClipper.'
            raise ValueError(msg)

        return self


@dataclass(frozen=True)
class EarlyStoppingConfig:
    """Specification for early stopping.

    Attributes:
        active: Whether to enable early stopping.
        window: Number of recent validation metrics to average.
        patience: Epochs to wait for improvement before stopping.
    """

    active: bool
    window: int
    patience: int


@dataclass(frozen=True)
class TrainingConfig:
    """Specification for the training loop.

    Attributes:
        batch_size: Number of samples in a batch.
        learn: Optimiser and scheduler configuration.
        n_epochs: Maximum number of training epochs.
        early_stopping: Early stopping configuration.
    """

    batch_size: StrictlyPositiveInt
    learn: LearningConfig
    n_epochs: StrictlyPositiveInt
    early_stopping: EarlyStoppingConfig


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SVCConfig:
    """Settings of the support vector classifier, as scikit-learn names them.

    Attributes:
        name: Support vector classifier.
        kernel: Kernel of the classifier.
        c: Regularization parameter ``C``.
        gamma: Kernel coefficient.
    """

    name: Literal[Classifiers.SVC]
    kernel: Literal['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']
    c: PositiveFloat
    gamma: float | Literal['scale', 'auto']


@dataclass(frozen=True)
class ClassifierArchitectureConfig:
    """Architecture of the MLP classifier, whose encoder the autoencoder pretrains.

    Attributes:
        name: MLP, or MLP with the pretrained encoder of the autoencoder.
        hidden_dims: Widths of the hidden layers of the encoder.
        z_dim: Width of the encoding.
    """

    name: Literal[Classifiers.MLP, Classifiers.AE]
    hidden_dims: tuple[StrictlyPositiveInt, ...]
    z_dim: StrictlyPositiveInt


@dataclass(frozen=True)
class PhaseConfig:
    """Specification for a single phase.

    Attributes:
        train: Training loop and learning schedule configuration.
    """

    train: TrainingConfig


@dataclass(frozen=True)
class ClassificationConfig(PhaseConfig):
    """Specification for the classifiers.

    Attributes:
        train: Training of the MLP classifiers.
        model: Classifier to train, whose probabilities condition the counterfactual VAE.
        rescale_steps: How many times the probabilities of each classifier are rescaled
            towards the middle before conditioning the VAE.
    """

    model: Annotated[ClassifierArchitectureConfig | SVCConfig, Field(discriminator='name')]
    rescale_steps: dict[Classifiers, PositiveInt]


@dataclass(frozen=True)
class AutoencoderModelConfig:
    """Autoencoder model, whose architecture is the one of the classifier.

    Attributes:
        name: Autoencoder architecture.
    """

    name: str


@dataclass(frozen=True)
class AutoencodingConfig(PhaseConfig):
    """Specification for the autoencoder pretraining the encoder of the ``ae`` classifier.

    Attributes:
        train: Training of the autoencoder.
        model: Autoencoder model.
    """

    model: AutoencoderModelConfig


@dataclass(frozen=True)
class VAEArchitectureConfig:
    """Architecture of the counterfactual VAE.

    Attributes:
        name: VAE architecture.
        hidden_dims: Widths of the hidden layers, one latent stage for each.
        z_dim: Width of each latent stage.
    """

    name: str
    hidden_dims: tuple[StrictlyPositiveInt, ...]
    z_dim: StrictlyPositiveInt


@dataclass(frozen=True)
class ObjectiveConfig:
    """Loss of the counterfactual VAE.

    Attributes:
        c_kld: Weight of the KL divergence against the reconstruction error.
    """

    c_kld: PositiveFloat


@dataclass(frozen=True)
class CounterfactualConfig(PhaseConfig):
    """Specification for the counterfactual VAE.

    Attributes:
        train: Training of the VAE.
        model: Architecture of the VAE.
        objective: Loss of the VAE.
    """

    model: VAEArchitectureConfig
    objective: ObjectiveConfig


# ---------------------------------------------------------------------------
# User / environment settings — generic
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrackerList:
    """Flags controlling which experiment trackers are active.

    Attributes:
        hydra: Saves a copy of the resolved config to the Hydra output dir.
        csv: Appends metrics to a CSV file in the experiment directory.
        tensorboard: Writes the metrics, the confusion matrices and the figures to TensorBoard logs.
        wandb: Uploads the metrics, the confusion matrices and the figures to Weights & Biases (requires
            ``track_all`` extra).
        sqlalchemy: Persists metrics in a SQLite database (requires ``track_all`` extra).
    """

    hydra: bool
    csv: bool
    tensorboard: bool
    wandb: bool
    sqlalchemy: bool


@dataclasses.dataclass(frozen=True)
class HydraSettings:
    """Subset of Hydra runtime settings, populated automatically at startup.

    Attributes:
        output_dir: Directory where Hydra stores logs and config dumps.
        job_logging: Hydra job-logging DictConfig used to configure Python logging.
        config_name: Name of the file in ``configs/experiment`` the configuration starts from.
    """

    output_dir: pathlib.Path = pathlib.Path('outputs')
    job_logging: Any = None
    config_name: str = ''


@dataclass(frozen=True)
class PathSpecs:
    """Filesystem path specifications populated from environment variables.

    Attributes:
        data_dir: Root directory holding the datasets, shared by all versions.
        root_exp_dir: Root directory under which all experiment artefacts are stored.

    Note:
        Paths are loaded from the ``.env`` file via :class:`~src.config.environment.EnvSettings`.
    """

    _env = EnvSettings()
    data_dir: pathlib.Path = _env.data_dir
    root_exp_dir: pathlib.Path = _env.root_exp_dir

    @property
    def version_dir(self) -> pathlib.Path:
        """Version-scoped experiment directory: ``<root_exp_dir>/v<VERSION>``."""
        return self.root_exp_dir / f'v{VERSION}'


@dataclass(frozen=True)
class PlottingSettings:
    """Options for visual inspection.

    Attributes:
        sample_indices: Subjects of the evaluation partition whose figures the visualization publishes.
    """

    sample_indices: list[PositiveInt]


@dataclass(frozen=True)
class UserSettings:
    """User-specific runtime preferences.

    Attributes:
        device: Device where the models run: ``auto`` for the accelerator when available, or a name such as
            ``cpu`` or ``cuda:1``.
        n_workers: DataLoader worker processes. ``0`` runs loading in the main process.
        trackers: Active experiment-tracking backends.
        seed: Global random seed for reproducibility. ``None`` leaves it unset.
        checkpoint_every: Save a model checkpoint every *n* epochs; ``0`` saves only at the end of training.
        torch_compile: Whether to compile the model with torch.compile.
        load: Checkpoint epoch to resume from:``0`` skip; ``-1`` loads last checkpoint.
        run_id: Run to resume. Defaults to the last run of the experiment.
        plot: Options for visual inspection.
    """

    device: Device
    n_workers: PositiveInt
    trackers: TrackerList
    seed: int | None
    checkpoint_every: PositiveInt
    torch_compile: bool
    load: int
    run_id: str | None
    plot: PlottingSettings
    hydra: HydraSettings = dataclasses.field(default_factory=HydraSettings)
    path: PathSpecs = dataclasses.field(default_factory=PathSpecs)

    def __post_init__(self) -> None:
        """Apply the random seed immediately after the config is constructed."""
        if self.seed is not None:
            set_seed(self.seed)

        return


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExperimentConfig:
    """Root configuration object for all experiment settings.

    Attributes:
        data: Dataset configuration.
        classification: Classifiers configuration.
        autoencoding: Autoencoder pretraining configuration.
        counterfactual: Counterfactual VAE configuration.
        user: User-specific options and runtime preferences.
        final: If True, uses validation data for training and test data for evaluation.
        variation: Variation name for the experiment (defaults to git branch if None).
        tags: List of tags describing active configuration overrides.
    """

    data: DataConfig
    classification: ClassificationConfig
    autoencoding: AutoencodingConfig
    counterfactual: CounterfactualConfig
    user: UserSettings
    final: bool
    variation: str | None = None
    tags: list[str] = dataclasses.field(default_factory=list)
    version: str = dataclasses.field(default=f'v{VERSION}', init=False)

    @cached_property
    def git_branch(self) -> str:
        """Current git branch (used when variation is unset)."""
        try:
            return get_current_branch()
        except ValueError as ve:
            msg = "Could not resolve git branch. Set 'variation' in the configuration file."
            raise ValueError(msg) from ve

    @property
    def resolved_variation(self) -> str:
        """variation if set, otherwise the current git branch."""
        return self.variation or self.git_branch

    @property
    def name(self) -> str:
        """Full experiment name embedded into tracker runs."""
        suffix = '_final' if self.final else ''
        return self.resolved_variation[:249] + suffix

    @property
    def par_dir(self) -> pathlib.Path:
        """Directory of the experiments of this configuration: ``<version_dir>/<config_name>``."""
        return self.user.path.version_dir / self.user.hydra.config_name

    @property
    def project(self) -> str:
        """Tracker project name, one per version."""
        return f'{PROJECT_NAME}-{self.version}'
