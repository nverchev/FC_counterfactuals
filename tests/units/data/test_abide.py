"""Tests for the ABIDE data builder."""

import dataclasses
import pathlib
from collections.abc import Generator

import pandas as pd
import pytest
import torch

from src.config import ExperimentConfig, Experiment
from src.data.abide import AbideDataBuilder
from src.data.split import Partitions

from src.data.abide import N_FEATURES, RESOLUTION

from tests.conftest import N_SUBJECTS


class TestAbideDataBuilder:
    """Test the partitions and the features of the ABIDE splits."""

    @pytest.fixture
    def databuilder(self) -> AbideDataBuilder:
        """Provide the builder of the fake dataset."""
        return AbideDataBuilder()

    def test_partitions_split_the_subjects(self, databuilder: AbideDataBuilder) -> None:
        """Training and validation make up the subjects that the test partition leaves."""
        ids = {partition: set(databuilder.split(partition).subject_ids) for partition in Partitions}

        assert ids[Partitions.train] | ids[Partitions.val] == ids[Partitions.train_val]
        assert not ids[Partitions.train] & ids[Partitions.val]
        assert not ids[Partitions.train_val] & ids[Partitions.test]
        assert len(ids[Partitions.train_val]) + len(ids[Partitions.test]) == N_SUBJECTS

    def test_features_are_the_connections_above_the_diagonal(self, databuilder: AbideDataBuilder) -> None:
        """Each subject has one feature for each pair of regions."""
        split = databuilder.split(Partitions.test)

        features = split.features

        assert features.shape == (len(split), N_FEATURES)
        assert features.dtype == torch.float32

    def test_labels_mark_typical_development_as_one(self, databuilder: AbideDataBuilder) -> None:
        """The diagnoses 1 (autism) and 2 (control) become 0 and 1."""
        split = databuilder.split(Partitions.train_val)
        metadata = pd.read_csv(databuilder.dataset_dir / 'metadata.csv').set_index('file')

        expected = torch.tensor([[float(metadata.loc[file, 'label']) - 1] for file in split.subject_ids])

        torch.testing.assert_close(split.labels, expected)


class TestAbideMetadata:
    """Test the metadata written from the phenotypes."""

    @pytest.fixture
    def phenotype_config(self, example_config: ExperimentConfig, tmp_path: pathlib.Path) -> ExperimentConfig:
        """Provide a configuration whose dataset folder holds only the phenotypes and the matrices."""
        source = example_config.user.path.data_dir / 'abide_fc_dataset'
        folder = tmp_path / 'abide_phenotypes'
        (folder / 'ABIDE_pcp').mkdir(parents=True)
        (folder / f'corr_matrices_{RESOLUTION}').symlink_to(source / f'corr_matrices_{RESOLUTION}')
        files = pd.read_csv(source / 'metadata.csv')['file']
        passed = ['OK'] * len(files)
        failed = ['fail'] + ['OK'] * (len(files) - 1)
        pd.DataFrame(
            {
                'FILE_ID': files,
                'DX_GROUP': 1,
                'qc_rater_1': failed,
                'qc_anat_rater_2': passed,
                'qc_func_rater_2': passed,
                'qc_anat_rater_3': passed,
                'qc_func_rater_3': passed,
            }
        ).to_csv(folder / 'ABIDE_pcp' / 'Phenotypic_V1_0b_preprocessed1.csv', index=False)
        dataset_cfg = dataclasses.replace(example_config.data.dataset, folder=str(folder))
        return dataclasses.replace(example_config, data=dataclasses.replace(example_config.data, dataset=dataset_cfg))

    @pytest.fixture
    def active_experiment(self, phenotype_config: ExperimentConfig) -> Generator[Experiment]:
        """Provide an active experiment reading the phenotypes, in place of the shared one."""
        exp = Experiment(phenotype_config, name='phenotypes', par_dir=phenotype_config.par_dir)
        with exp.create_run(record=False):
            yield exp

        return

    def test_metadata_keeps_the_subjects_passing_the_checks(self) -> None:
        """A subject failing a quality check is left out of the written metadata."""
        databuilder = AbideDataBuilder()

        metadata = pd.read_csv(databuilder.dataset_dir / 'metadata.csv')

        assert list(metadata.columns) == ['file', 'label']
        assert len(metadata) == N_SUBJECTS - 1
