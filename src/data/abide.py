"""Functional connectomes of the ABIDE preprocessed dataset."""

import logging
import pathlib
import urllib.request
import zipfile
from typing import Final, override

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from src.config.experiment import Experiment
from src.data.databuilder import DataBuilder
from src.data.split import DataSplit, Partitions

__all__ = ['N_FEATURES', 'RESOLUTION', 'URL', 'AbideDataBuilder']

RESOLUTION: Final = 122
"""Number of regions of the atlas, the side of each connectivity matrix."""

N_FEATURES: Final = RESOLUTION * (RESOLUTION - 1) // 2
"""Number of connections above the diagonal of a connectivity matrix."""

URL: Final = 'https://gaimfs.ugent.be/Public/Dataset/abide_fc_dataset.zip'
"""Archive of the dataset, downloaded when its folder is missing."""

_PHENOTYPES: Final = pathlib.Path('ABIDE_pcp') / 'Phenotypic_V1_0b_preprocessed1.csv'


class AbideDataBuilder(DataBuilder):
    """Correlation matrices of the ABIDE subjects that pass the quality checks.

    The folder holds ``metadata.csv``, with the identifier and the diagnosis of each subject,
    and a ``corr_matrices_<resolution>`` folder with the correlation matrix of each subject.
    A missing folder is downloaded; a missing ``metadata.csv`` is written from the phenotypes.

    Attributes:
        dataset_dir: Folder of the dataset.
        test_fraction: Fraction of the subjects held out for the test partition.
        val_fraction: Fraction of the remaining subjects held out for the validation partition.
        split_seed: Seed of the random partitioning.
    """

    dataset_dir: Final[pathlib.Path]
    test_fraction: Final[float]
    val_fraction: Final[float]
    split_seed: Final[int]
    _metadata: dict[Partitions, pd.DataFrame]

    def __init__(self) -> None:
        """Initialize."""
        cfg_dataset = Experiment.get_config().data.dataset
        self.dataset_dir = Experiment.get_config().user.path.data_dir / cfg_dataset.folder
        self.test_fraction = cfg_dataset.test_fraction
        self.val_fraction = cfg_dataset.val_fraction
        self.split_seed = cfg_dataset.split_seed
        self._metadata = {}
        super().__init__()
        return

    @property
    def matrices_dir(self) -> pathlib.Path:
        """Folder of the correlation matrices."""
        return self.dataset_dir / f'corr_matrices_{RESOLUTION}'

    @override
    def split(self, partition: Partitions, /) -> DataSplit:
        metadata = self._metadata[partition]
        upper = np.triu_indices(RESOLUTION, k=1)
        features = np.stack([np.load(self.matrices_dir / f'{file}.npy')[upper] for file in metadata['file']])
        labels = metadata['label'].to_numpy() - 1  # the phenotypes write 1 for autism and 2 for controls
        return DataSplit(
            torch.from_numpy(features).float(),
            torch.from_numpy(labels).float().unsqueeze(1),
            subject_ids=tuple(metadata['file']),
        )

    @override
    def _download(self) -> None:
        if self.dataset_dir.exists():
            return

        archive = self.dataset_dir.with_suffix('.zip')
        logging.info('Downloading the dataset from %s.', URL)
        urllib.request.urlretrieve(URL, archive)
        with zipfile.ZipFile(archive) as zip_file:
            zip_file.extractall(self.dataset_dir.parent)

        return

    @override
    def _preprocess(self) -> None:
        metadata_path = self.dataset_dir / 'metadata.csv'
        if not metadata_path.exists():
            self._write_metadata(metadata_path)

        if not self.matrices_dir.exists():
            msg = f'No correlation matrices at resolution {RESOLUTION} in {self.dataset_dir}.'
            raise FileNotFoundError(msg)

        metadata = pd.read_csv(metadata_path)[['file', 'label']]
        positions = np.arange(len(metadata))
        train_val, test = train_test_split(positions, test_size=self.test_fraction, random_state=self.split_seed)
        train, val = train_test_split(train_val, test_size=self.val_fraction, random_state=self.split_seed)
        self._metadata = {
            Partitions.train: metadata.iloc[train],
            Partitions.train_val: metadata.iloc[train_val],
            Partitions.val: metadata.iloc[val],
            Partitions.test: metadata.iloc[test],
        }
        return

    def _write_metadata(self, path: pathlib.Path, /) -> None:
        phenotypes = pd.read_csv(self.dataset_dir / _PHENOTYPES)
        quality_checks = [
            phenotypes['qc_rater_1'] == 'OK',
            phenotypes['qc_anat_rater_2'].isin(['OK', 'maybe']),
            phenotypes['qc_func_rater_2'].isin(['OK', 'maybe']),
            phenotypes['qc_anat_rater_3'] == 'OK',
            phenotypes['qc_func_rater_3'] == 'OK',
        ]
        passed = phenotypes.loc[np.logical_and.reduce(quality_checks)]
        metadata = passed[['FILE_ID', 'DX_GROUP']].rename(columns={'FILE_ID': 'file', 'DX_GROUP': 'label'})
        metadata.to_csv(path, index=False)
        return
