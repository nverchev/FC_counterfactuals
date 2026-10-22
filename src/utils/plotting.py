"""Figures of connectivity matrices and connectomes."""

import functools
import pathlib

import matplotlib.pyplot as plt
import nilearn.plotting as nip
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from nilearn.image import load_img

from src.config.experiment import Experiment
from src.data.abide import RESOLUTION
from src.lib.events import FigureEvent

__all__ = [
    'LABEL_NAMES',
    'describe_probability',
    'publish_connectome',
    'to_matrix',
]

EDGE_THRESHOLD = 0.7
"""Smallest absolute correlation drawn as an edge of the connectome."""

LABEL_NAMES = ('autism spectrum disorder', 'typical development')
"""Names of the diagnoses, by label."""


def describe_probability(probability: float, /) -> str:
    """Describe the prediction of the classifier, as in ``predicted typical development (0.82)``.

    Args:
        probability: the probability of typical development the classifier assigns.

    Returns:
        The predicted diagnosis and its probability.
    """
    label = int(probability >= 0.5)
    return f'predicted {LABEL_NAMES[label]} ({max(probability, 1 - probability):.2f})'


def publish_connectome(features: torch.Tensor, /, *, title: str, caption: str, folder: str) -> None:
    """Publish the correlation matrix and the connectome of a subject.

    Args:
        features: connections of the subject.
        title: the name of the figure.
        caption: the text above the figure.
        folder: the subfolder of the run's figures.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(caption)
    matrix = to_matrix(features)
    _plot_heatmap(axes[0], matrix, 'Correlations')
    _plot_connectome(fig, axes[1], matrix)
    FigureEvent(figure=fig, title=title, folder=folder)
    plt.close(fig)
    return


def to_matrix(features: torch.Tensor, /) -> np.ndarray:
    """Rebuild the symmetric correlation matrix from the connections above its diagonal.

    Args:
        features: connections above the diagonal, in row order.

    Returns:
        The matrix, with ones on the diagonal.
    """
    matrix = np.zeros((RESOLUTION, RESOLUTION))
    matrix[np.triu_indices(RESOLUTION, k=1)] = features.detach().cpu().numpy()
    matrix += matrix.T
    matrix[np.diag_indices(RESOLUTION)] = 1
    return matrix


def _plot_connectome(fig: Figure, ax: Axes, matrix: np.ndarray) -> None:
    cfg = Experiment.get_config()
    cfg_dataset = cfg.data.dataset
    atlas_dir = cfg.user.path.data_dir / cfg_dataset.folder / 'ABIDE_pcp' / 'basc_multiscale_2015'
    atlas_dir /= 'template_cambridge_basc_multiscale_nii_sym'
    coords = _get_region_coordinates(atlas_dir / f'template_cambridge_basc_multiscale_sym_scale{RESOLUTION:03d}.nii.gz')

    # nilearn fails to draw the colour bar when no edge passes the threshold
    has_edges = bool((np.abs(matrix[np.triu_indices(len(matrix), k=1)]) > EDGE_THRESHOLD).any())
    nip.plot_connectome(
        matrix,
        coords,
        display_mode='lzry',
        edge_threshold=EDGE_THRESHOLD,
        node_size=3,
        colorbar=has_edges,
        figure=fig,
        axes=ax,
    )
    return


@functools.cache
def _get_region_coordinates(atlas_path: pathlib.Path) -> np.ndarray:
    return np.asarray(nip.find_parcellation_cut_coords(load_img(atlas_path)))


def _plot_heatmap(ax: Axes, matrix: np.ndarray, title: str) -> None:
    ax.imshow(matrix, cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_title(title, fontsize=18)
    return
