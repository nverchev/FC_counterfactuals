import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from nilearn import datasets
from nilearn.connectome import ConnectivityMeasure
from nilearn.maskers import NiftiLabelsMasker


def fetch_preprocess(data_dir):
    os.mkdir(data_dir)
    dataset = datasets.fetch_abide_pcp(data_dir=data_dir, legacy_format=False)
    atlas = datasets.fetch_atlas_basc_multiscale_2015(version='sym', data_dir=data_dir)
    atlas = atlas.scale444
    masker = NiftiLabelsMasker(
        labels_img=atlas,
        standardize=True,
        memory='nilearn_cache',
        verbose=1)

    corr_dir = os.path.join(data_dir, 'corr_matrices')

    for i in range(len(dataset.func_preproc)):
        print(f'Processing {i} of {len(dataset.func_preproc)}')
        file_id = dataset.phenotypic.iloc[i]['FILE_ID']
        time_series = masker.fit_transform(dataset.func_preproc[i])
        correlation_measure = ConnectivityMeasure(kind='correlation')
        correlation_matrix = correlation_measure.fit_transform([time_series])[0]
        np.save(os.path.join(corr_dir, f'{file_id}.npy'), correlation_matrix)




class EmptyDataset(Dataset):

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return NotImplementedError


class FCSplitDataset(Dataset):
    def __init__(self, metadata, data_dir, **dataset_settings):
        self.input_files = metadata['file'].values
        # TODO: use the same labels everywhere
        self.labels = metadata['label'].values - 1  # 1 -> 0, 2 -> 1
        self.matrices = np.zeros(shape=(len(self.labels), 98346))
        matrices_path = os.path.join(data_dir, 'corr_matrices')
        for idx, file in enumerate(self.input_files):
            matrix = np.load(os.path.join(matrices_path, file))
            # TODO: store triu directly
            # FIXME: dynamically change 98346 and 444 depending on the matrix resolution
            x = matrix[np.triu_indices(444, k=1)]
            self.matrices[idx] = x

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.matrices[idx]).float()
        label = torch.tensor([self.labels[idx]]).float()
        return x, label, idx


class FCDataset:
    def __init__(self, data_dir, **dataset_settings):
        self.data_dir = data_dir
        self.dataset_settings = dataset_settings
        # TODO: get metadata
        # metadata = pd.read_csv(os.path.join(data_dir, 'ABIDE_pcp', 'Phenotypic_V1_0b_preprocessed1.csv'))
        # TODO: dataset partition logic here
        self.partition()

    def split(self, split):
        # TODO: more info than just labels (probabilities from classifier for example)
        metadata_path = os.path.join(self.data_dir, f'labels_{split}.csv')
        metadata = pd.read_csv(metadata_path)

        return FCSplitDataset(metadata, self.data_dir, **self.dataset_settings)

    def partition(self):
        return


def get_loaders(batch_size, data_dir, **dataset_settings):
    if not os.path.exists(data_dir):
        fetch_preprocess(data_dir)

    pin_memory = torch.cuda.is_available()

    dataset = FCDataset(data_dir=data_dir, **dataset_settings)
    train_dataset = dataset.split('train')
    # FIXME: only labels_train and labels_test
    val_dataset = EmptyDataset()
    test_dataset = dataset.split('test')

    train_loader = torch.utils.data.DataLoader(
        train_dataset, drop_last=True, batch_size=batch_size, shuffle=True, pin_memory=pin_memory)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, drop_last=False, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, drop_last=False, shuffle=False, pin_memory=pin_memory)

    return train_loader, val_loader, test_loader
