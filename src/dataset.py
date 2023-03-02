import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from nilearn import datasets
from nilearn.connectome import ConnectivityMeasure
from nilearn.maskers import NiftiLabelsMasker


def fetch_preprocess(data_dir, res):
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
        dataset = datasets.fetch_abide_pcp(data_dir=data_dir, legacy_format=False)
        metadata_df = pd.read_csv(os.path.join(data_dir, 'ABIDE_pcp', 'Phenotypic_V1_0b_preprocessed1.csv'))
        quality_checks = [
            metadata_df['qc_rater_1'] == 'OK',
            metadata_df['qc_anat_rater_2'].isin(['OK', 'maybe']),
            metadata_df['qc_func_rater_2'].isin(['OK', 'maybe']),
            metadata_df['qc_anat_rater_3'] == 'OK',
            metadata_df['qc_func_rater_3'] == 'OK'
        ]
        metadata_df = metadata_df.loc[np.logical_and.reduce(quality_checks)]
        metadata_df = metadata_df[['FILE_ID', 'DX_GROUP']].rename(columns={'FILE_ID': 'file', 'DX_GROUP': 'label'})
        metadata_df['svc_prob'] = metadata_df['mlp_prob'] = metadata_df['ae_prob'] = np.nan
        metadata_df.to_csv(os.path.join(data_dir, 'metadata.csv'))

    corr_dir = os.path.join(data_dir, 'corr_matrices_' + str(res))
    if not os.path.exists(corr_dir):
        os.mkdir(corr_dir)
        metadata_df = pd.read_csv(os.path.join(data_dir, 'metadata.csv'))
        atlas = datasets.fetch_atlas_basc_multiscale_2015(version='sym', data_dir=data_dir)
        # TODO: choose matrix resolution, maybe add more choices to options
        atlas = atlas.scale444
        masker = NiftiLabelsMasker(
            labels_img=atlas,
            standardize=True,
            memory='nilearn_cache',
            verbose=1)

        # FIXME: variable "dataset" is not referenced when the dataset
        #  has already been downloaded but not processed with new res
        for i, file_id in enumerate(metadata_df['file']):
            print(f'Processing {i} of {len(metadata_df)}')
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
    def __init__(self, metadata, data_dir, res, cond='svc', **dataset_settings):
        self.metadata = metadata
        # TODO: use the same labels everywhere
        self.labels = metadata['label'].values - 1  # 1 -> 0, 2 -> 1
        self.matrices = np.zeros(shape=(len(self.labels), res * (res - 1) // 2))
        self.condition = metadata[cond + '_prob'].values
        matrices_path = os.path.join(data_dir, 'corr_matrices_' + str(res))
        for idx, file in enumerate(self.metadata['file']):
            matrix = np.load(os.path.join(matrices_path, file + '.npy'))
            # TODO: store triu directly ?
            x = matrix[np.triu_indices(res, k=1)]
            self.matrices[idx] = x

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.matrices[idx]).float()
        label = torch.tensor([self.labels[idx]]).float()
        condition = torch.tensor([self.condition[idx]]).float()

        return [x, condition], label, idx


class FCDataset:
    def __init__(self, data_dir, res, **dataset_settings):
        self.data_dir = data_dir
        self.dataset_settings = dataset_settings
        self.res = res
        metadata_df = pd.read_csv(os.path.join(data_dir, 'metadata.csv'))
        train_val_df, test_df = train_test_split(metadata_df, test_size=0.2, random_state=2023)
        train_df, val_df = train_test_split(train_val_df, test_size=0.2, random_state=2023)
        self.partitions = {
            'train_val': train_val_df,
            'test': test_df,
            'train': train_df,
            'val': val_df
        }

    def split(self, split):
        metadata = self.partitions[split]
        return FCSplitDataset(metadata, self.data_dir, self.res, **self.dataset_settings)

    def partition(self):
        return


def get_loaders(data_dir, res, batch_size, exp, **other_settings):
    dataset_settings = dict(
        res=res
    )

    fetch_preprocess(data_dir, res)
    pin_memory = torch.cuda.is_available()
    dataset = FCDataset(data_dir=data_dir, **dataset_settings)
    if exp[:5] == 'final':
        train_dataset = dataset.split('train_val')
        val_dataset = EmptyDataset()
        test_dataset = dataset.split('test')
    else:
        train_dataset = dataset.split('train')
        val_dataset = dataset.split('val')
        test_dataset = EmptyDataset()

    train_loader = torch.utils.data.DataLoader(
        train_dataset, drop_last=False, batch_size=batch_size, shuffle=True, pin_memory=pin_memory)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, drop_last=False, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, drop_last=False, shuffle=False, pin_memory=pin_memory)

    return train_loader, val_loader, test_loader
