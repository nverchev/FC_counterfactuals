"""Shared fixtures: a small fake ABIDE dataset and an experiment reading it."""

import dataclasses
import pathlib
from collections.abc import Generator

import numpy as np
import pandas as pd
import pytest

from src.config import ExperimentConfig, Experiment, get_experiment_config
from src.data.abide import RESOLUTION
from src.config.specs import PathSpecs

N_SUBJECTS = 20


@pytest.fixture(scope='package')
def example_data_dir(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    """Provide a data directory holding a fake ABIDE dataset."""
    root = tmp_path_factory.mktemp('data')
    matrices_dir = root / 'abide_fc_dataset' / f'corr_matrices_{RESOLUTION}'
    matrices_dir.mkdir(parents=True)
    rng = np.random.default_rng(0)
    files = [f'SITE_{i:07d}' for i in range(N_SUBJECTS)]
    labels = [1 + i % 2 for i in range(N_SUBJECTS)]
    pd.DataFrame({'file': files, 'label': labels}).to_csv(matrices_dir.parent / 'metadata.csv', index=False)
    for file in files:
        values = rng.uniform(-1, 1, (RESOLUTION, RESOLUTION))
        matrix = (values + values.T) / 2
        np.fill_diagonal(matrix, 1)
        np.save(matrices_dir / f'{file}.npy', matrix)

    return root


@pytest.fixture(scope='package')
def example_config(example_data_dir: pathlib.Path) -> ExperimentConfig:
    """Provide a configuration reading the fake dataset, with small models on the CPU."""
    overrides = [
        'classification.model.hidden_dims=[16,8]',
        'classification.model.z_dim=4',
        'counterfactual.model.hidden_dims=[16,8]',
        'counterfactual.model.z_dim=4',
        'user.device=cpu',
    ]
    cfg = get_experiment_config('mlp', overrides=overrides)
    path = PathSpecs(data_dir=example_data_dir, root_exp_dir=example_data_dir / 'experiments')
    return dataclasses.replace(cfg, user=dataclasses.replace(cfg.user, path=path))


@pytest.fixture(autouse=True)
def active_experiment(example_config: ExperimentConfig) -> Generator[Experiment]:
    """Provide an active experiment run, which the code reads its configuration from."""
    exp = Experiment(example_config, name='test_exp', par_dir=example_config.par_dir)
    with exp.create_run(record=False):
        yield exp

    return
