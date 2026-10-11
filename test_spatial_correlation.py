"""Is the reconstruction noise spatially uncorrelated?

For a connectome, the analogue of "neighbouring pixels" is a pair of edges that
are close in the brain: edges sharing an endpoint region, or edges whose
endpoints sit close together in MNI space.  Blur-type reconstruction error would
show correlation that decays with that distance; unstructured error would not.

Reports, for the residual and for the original data:
  - mean |corr| between edge pairs that share a node vs disjoint pairs
  - mean |corr| as a function of MNI distance between the edges
  - a column-shuffled null giving the finite-sample floor

Usage: python test_spatial_correlation.py [--cond svc] [--part train] [--n_edges 2500]
"""
import argparse
import os
import warnings

import numpy as np
from nilearn.image import load_img
from nilearn import plotting as nip
from scipy.spatial.distance import cdist

from src.independence_utils import get_partition, load_setup, reconstruct

warnings.filterwarnings('ignore')


def atlas_coords(data_dir, res):
    path = os.path.join(data_dir, 'ABIDE_pcp', 'basc_multiscale_2015',
                        'template_cambridge_basc_multiscale_nii_sym',
                        f'template_cambridge_basc_multiscale_sym_scale{res}.nii.gz')
    coords, _ = nip.find_parcellation_cut_coords(load_img(path), return_label_names=True)
    return np.asarray(coords)


def edge_pair_distance(coords, e):
    """(N,N) distance between edges: best matching of their endpoints."""
    c0, c1 = coords[e[:, 0]], coords[e[:, 1]]
    d00, d11 = cdist(c0, c0), cdist(c1, c1)
    d01, d10 = cdist(c0, c1), cdist(c1, c0)
    return np.minimum(np.maximum(d00, d11), np.maximum(d01, d10))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cond', nargs='+', default=['svc', 'mlp', 'ae'])
    ap.add_argument('--part', default='train')
    ap.add_argument('--n_edges', type=int, default=2500)
    ap.add_argument('--seed', type=int, default=2023)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    for cond in a.cond:
        trainer, clf, args = load_setup(cond)
        coords = atlas_coords(args.data_dir, args.res)
        rows, cols = np.triu_indices(args.res, k=1)
        edges = np.stack([rows, cols], 1)

        X, y, p = get_partition(trainer, a.part)
        r = X - reconstruct(trainer, X, p)
        n = len(y)
        idx = rng.choice(len(edges), a.n_edges, replace=False)
        e = edges[idx]

        shares = ((e[:, None, 0] == e[None, :, 0]) | (e[:, None, 0] == e[None, :, 1]) |
                  (e[:, None, 1] == e[None, :, 0]) | (e[:, None, 1] == e[None, :, 1]))
        dist = edge_pair_distance(coords, e)
        off = ~np.eye(a.n_edges, dtype=bool)

        print(f'\n{"=" * 70}\ncond={cond}  partition={a.part}  n={n}  '
              f'edges sampled={a.n_edges}\n{"=" * 70}')
        shuffled = np.stack([rng.permutation(r[:, k]) for k in idx], 1)
        for name, M in [('residual r', r[:, idx]), ('original x', X[:, idx]),
                        ('null (shuffled)', shuffled)]:
            C = np.corrcoef((M - M.mean(0)).T)
            ac = np.abs(C)
            sh, ds = shares & off, (~shares) & off
            print(f'\n  {name}')
            print(f'    shares a node : mean|corr|={ac[sh].mean():.4f}   (n pairs={sh.sum():,})')
            print(f'    disjoint      : mean|corr|={ac[ds].mean():.4f}   (n pairs={ds.sum():,})')
            print(f'    difference    : {ac[sh].mean() - ac[ds].mean():+.4f}')
            bins = [0, 20, 40, 60, 80, 200]
            out = []
            for lo, hi in zip(bins[:-1], bins[1:]):
                m = ds & (dist >= lo) & (dist < hi)
                out.append(f'{lo:>3}-{hi:<3}mm: {ac[m].mean():.4f}' if m.sum() > 500 else
                           f'{lo:>3}-{hi:<3}mm:   n/a')
            print('    by MNI distance (disjoint pairs only)   ' + '   '.join(out))


if __name__ == '__main__':
    main()
