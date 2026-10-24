"""Test the assumption required by the construction of the counterfactuals.

A counterfactual applies a generated change to the original subject while preserving the
residual, the part the VAE does not reconstruct:

    counterfactual = clip(original + change)
    change = dec(z*, p*) - dec(z, p),  residual = original - dec(z, p)

The assumption: the residual is uncorrelated with the change. The residual may carry class
information, but it must not align with the direction of the change; otherwise, preserving it
weakens or inflates the effect the metrics measure.

  G1  geometry: cosine between the residual and the change, tested against residuals donated
      by other subjects, and whether the alignment depends on the class
  G2  scale: component of the residual along the change, as a fraction of the change
      (-1 = the residual cancels the change entirely)
  G3  function: is the classifier's response to the change the same with the residual,
      without it, and with a donated residual?

Usage::

    uv run analysis_recon_orthogonality.py
    uv run analysis_recon_orthogonality.py --classifiers svc mlp --n_null 200
"""

import argparse
import dataclasses
import pathlib

from collections.abc import Sequence
from typing import Final

import numpy as np
import torch

from scipy import stats

from src.config import ExperimentConfig, Experiment, get_experiment_config
from src.data import get_splits
from src.data.derived_datasets import ConditionedSplit, load_classifier
from src.module import CounterfactualVAE
from src.utils.analysis import AnalysisSetup

BANNER_WIDTH: Final = 78
COLUMN_GAP: Final = ' ' * 2
EPS: Final = 1e-6
INDENT: Final = ' ' * 4
LEGEND: Final = {
    'cos': 'cosine between the residual and the change (0 under H0; compare to null)',
    'along': 'component of the residual along the change, in units of the change '
    '(0 under H0; -1 = the residual cancels the change)',
    'attenuat.': 'mean effect of the change on the classifier with the residual / without it (1 under H0)',
    'corr w/wo': 'per-subject correlation of those two effects (1 under H0)',
    'corr(y,cos)': "does the alignment depend on the subject's class? (0 under H0)",
}
SUMMARY_COLUMNS: Final = (
    'cond',
    'partition',
    'target',
    'cos',
    'null',
    'along',
    'attenuat.',
    'corr w/wo',
    'corr(y,cos)',
)


@dataclasses.dataclass(frozen=True)
class Summary:
    """Summary of the tests for a classifier, a partition and a target.

    Attributes:
        classifier: the conditioning classifier.
        partition: the partition of the subjects.
        target: the target of the counterfactuals.
        cosine: mean cosine between the residual and the change.
        null: mean cosine with residuals donated by other subjects.
        along: mean component of the residual along the change, in units of the change.
        attenuation: mean effect of the change with the residual, divided by the one without it.
        correlation: per-subject correlation of the effects with and without the residual.
        class_correlation: point-biserial correlation of the label with the cosine.
    """

    classifier: str
    partition: str
    target: str
    cosine: float
    null: float
    along: float
    attenuation: float
    correlation: float
    class_correlation: float


def analyse(classifier: str, overrides: Sequence[str], n_null: int, seed: int) -> list[Summary]:
    """Run the tests for a classifier, on both partitions and for both targets.

    Args:
        classifier: the conditioning classifier: ``svc``, ``mlp`` or ``ae``.
        overrides: the Hydra overrides the models were trained with.
        n_null: the number of residuals donated to each subject for the null.
        seed: the seed of the donations.

    Returns:
        The summary of each partition and target.

    Side Effects:
        Prints the tests.
    """
    setup = load_setup(classifier, overrides)
    rng = np.random.default_rng(seed)
    summaries: list[Summary] = []
    for partition, description in [('train', 'train_val (seen by the VAE)'), ('test', 'test (held out)')]:
        features, labels, conditions = setup.partitions[partition]
        targets = {'opposite': np.less(conditions, 0.5).astype(np.float32), 'half': np.full_like(conditions, 0.5)}
        for target, target_probabilities in targets.items():
            _print_banner(f'classifier={classifier} | {description} | target={target} | n={len(labels)}')
            _, recon, recon_target = setup.counterfactuals(features, conditions, target_probabilities)
            residual = features - recon
            change = recon_target - recon
            cosine, null, class_correlation = _test_alignment(residual, change, labels, rng, n_null)
            along = _test_scale(residual, change)
            attenuation, correlation = _test_classifier(setup, features, recon, residual, change, rng, along)
            summary = Summary(
                classifier=classifier,
                partition=partition,
                target=target,
                cosine=cosine,
                null=null,
                along=float(along.mean()),
                attenuation=attenuation,
                correlation=correlation,
                class_correlation=class_correlation,
            )
            summaries.append(summary)

    return summaries


def load_setup(classifier: str, overrides: Sequence[str] = ('final=true',)) -> AnalysisSetup:
    """Load the trained counterfactual VAE conditioned on a classifier.

    Args:
        classifier: the conditioning classifier, which names the entry configuration: ``svc``, ``mlp`` or ``ae``.
        overrides: the Hydra overrides the models were trained with.

    Returns:
        The models and the partitions of the experiment.

    Raises:
        RuntimeError: if the experiment holds no trained models.
    """
    cfg = get_experiment_config(classifier, overrides=list(overrides))
    exp = Experiment(cfg, name=cfg.name, par_dir=cfg.par_dir)
    with exp.create_run(resume=True, record=False):
        trained_classifier = load_classifier()
        train_split, test_split = get_splits()
        splits = {
            'train': ConditionedSplit(train_split, trained_classifier),
            'test': ConditionedSplit(test_split, trained_classifier),
        }

        # drytorch numbers a repeated model name in a process, so the weights are read directly
        module = CounterfactualVAE()
        module.load_state_dict(torch.load(_get_last_state(cfg), map_location='cpu', weights_only=True))

    partitions = {
        name: (split.features.numpy(), split.labels.squeeze(1).long().numpy(), split.conditions.squeeze(1).numpy())
        for name, split in splits.items()
    }
    return AnalysisSetup(module=module.eval(), probability=trained_classifier.probability, partitions=partitions)


def main() -> None:
    """Parse the arguments, run the tests for each classifier and print the summary."""
    parser = argparse.ArgumentParser(description='Test whether the residual is aligned with the counterfactual change.')
    parser.add_argument('--classifiers', nargs='+', default=['svc', 'mlp', 'ae'], choices=['svc', 'mlp', 'ae'])
    parser.add_argument(
        '--overrides', nargs='*', default=['final=true'], help='Hydra overrides the models were trained with'
    )
    parser.add_argument('--n_null', type=int, default=20, help='residuals donated to each subject for the null')
    parser.add_argument('--seed', type=int, default=2023)
    args = parser.parse_args()
    summaries = [
        summary
        for classifier in args.classifiers
        for summary in analyse(classifier, args.overrides, args.n_null, args.seed)
    ]
    _print_summaries(summaries)
    return


def _cosine(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
    return np.sum(first * second, axis=1) / (norms + EPS)


def _get_derangement(n: int, rng: np.random.Generator) -> np.ndarray:
    # a permutation that moves every subject, so that each one receives another's residual
    permutation = rng.permutation(n)
    while np.any(permutation == np.arange(n)):
        permutation = rng.permutation(n)

    return permutation


def _get_last_state(cfg: ExperimentConfig) -> pathlib.Path:
    # the checkpoints are in <run name>/<date>/<time>/<model name>/epoch_<epoch>
    model_dirs = sorted((cfg.par_dir / 'checkpoints' / cfg.name).glob(f'*/*/{cfg.counterfactual.model.name}'))
    epoch_dirs = list(model_dirs[-1].glob('epoch_*')) if model_dirs else []
    if not epoch_dirs:
        msg = f"No trained counterfactual VAE in the experiment '{cfg.name}': train it first."
        raise RuntimeError(msg)

    return max(epoch_dirs, key=lambda path: int(path.name.removeprefix('epoch_'))) / 'model_state.pt'


def _logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities.astype(np.float64), EPS, 1 - EPS)
    return np.log(clipped / (1 - clipped))


def _print_banner(text: str) -> None:
    rule = '=' * BANNER_WIDTH
    print(f'\n{rule}\n{text}\n{rule}', flush=True)
    return


def _print_section(title: str, items: dict[str, str]) -> None:
    width = max(len(label) for label in items)
    print(f'\n{title}')
    for label, value in items.items():
        print(f'{INDENT}{label.ljust(width)}: {value}')

    return


def _print_summaries(summaries: Sequence[Summary]) -> None:
    _print_banner('SUMMARY | H0: the residual is uncorrelated with the counterfactual change')
    rows = [
        (
            summary.classifier,
            summary.partition,
            summary.target,
            f'{summary.cosine:+.4f}',
            f'{summary.null:+.4f}',
            f'{summary.along:+.3f}',
            f'{summary.attenuation:+.3f}',
            f'{summary.correlation:+.3f}',
            f'{summary.class_correlation:+.3f}',
        )
        for summary in summaries
    ]
    _print_table(SUMMARY_COLUMNS, rows)
    _print_section('Legend', LEGEND)
    return


def _print_table(columns: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    widths = [max(len(cell) for cell in column) for column in zip(columns, *rows, strict=True)]
    lines = [
        COLUMN_GAP.join(cell.rjust(width) for cell, width in zip(row, widths, strict=True)) for row in [columns, *rows]
    ]
    print(lines[0])
    print('-' * len(lines[0]))
    print('\n'.join(lines[1:]))
    return


def _test_alignment(
    residual: np.ndarray, change: np.ndarray, labels: np.ndarray, rng: np.random.Generator, n_null: int
) -> tuple[float, float, float]:
    n = len(labels)
    cosine = _cosine(residual, change)
    null = np.concatenate([_cosine(residual[_get_derangement(n, rng)], change) for _ in range(n_null)])
    null_p = stats.ttest_1samp(cosine, null.mean())[1]
    effect_size = (cosine.mean() - null.mean()) / cosine.std()
    by_class_p = stats.mannwhitneyu(cosine[labels == 0], cosine[labels == 1])[1]
    by_class = f'ASD={cosine[labels == 0].mean():+.4f}, control={cosine[labels == 1].mean():+.4f}'
    class_correlation = float(np.corrcoef(labels, cosine)[0, 1])
    class_correlation_p = stats.pointbiserialr(labels, cosine)[1]
    items = {
        'cosine between the residual and the change': f'mean={cosine.mean():+.4f}, sd={cosine.std():.4f}',
        'donated-residual null': f'mean={null.mean():+.4f}, sd={null.std():.4f}',
        'observed vs null mean': f't-test p={null_p:.2e}, Cohen d={effect_size:+.2f}',
        'by class': f'{by_class}, Mann-Whitney p={by_class_p:.2e}',
        'point-biserial corr(label, cos)': f'{class_correlation:+.3f}, p={class_correlation_p:.2e}',
    }
    _print_section('[G1] alignment of the residual with the change', items)
    return float(cosine.mean()), float(null.mean()), class_correlation


def _test_classifier(
    setup: AnalysisSetup,
    features: np.ndarray,
    recon: np.ndarray,
    residual: np.ndarray,
    change: np.ndarray,
    rng: np.random.Generator,
    along: np.ndarray,
) -> tuple[float, float]:
    classify = setup.classify
    donated = residual[_get_derangement(len(features), rng)]
    effect_without = _logit(classify(np.clip(recon + change, -1, 1))) - _logit(classify(np.clip(recon, -1, 1)))
    effect_with = _logit(classify(np.clip(features + change, -1, 1))) - _logit(classify(features))
    with_donated = _logit(classify(np.clip(recon + donated + change, -1, 1)))
    effect_donated = with_donated - _logit(classify(np.clip(recon + donated, -1, 1)))
    clipped = np.mean(np.abs(features + change) > 1)
    attenuation = effect_with.mean() / effect_without.mean() if effect_without.mean() else np.nan
    correlation = np.corrcoef(effect_with, effect_without)[0, 1]
    interaction = effect_with - effect_without
    paired_p = stats.ttest_rel(effect_with, effect_without)[1]
    spearman, spearman_p = stats.spearmanr(along, interaction)
    items = {
        'effect without the residual': f'mean={effect_without.mean():+.3f}, sd={effect_without.std():.3f}',
        'effect with the residual': f'mean={effect_with.mean():+.3f}, sd={effect_with.std():.3f}',
        'effect with a donated residual': f'mean={effect_donated.mean():+.3f}, sd={effect_donated.std():.3f}',
        'attenuation with/without': f'{attenuation:+.3f}',
        'corr(with, without)': f'{correlation:+.3f}',
        'paired difference (with - without)': f'mean={interaction.mean():+.3f}, p={paired_p:.2e}',
        'Spearman(residual along the change, interaction)': f'{spearman:+.3f}, p={spearman_p:.2e}',
    }
    _print_section(f'[G3] classifier response to the same change ({clipped:.2%} of entries clipped)', items)
    return float(attenuation), float(correlation)


def _test_scale(residual: np.ndarray, change: np.ndarray) -> np.ndarray:
    change_norm = np.linalg.norm(change, axis=1)
    residual_norm = np.linalg.norm(residual, axis=1)
    along = np.sum(residual * change, axis=1) / (change_norm**2 + EPS)
    low, high = stats.t.interval(0.95, len(along) - 1, loc=along.mean(), scale=stats.sem(along))
    items = {
        'along': f'mean={along.mean():+.3f}, 95% CI=[{low:+.3f}, {high:+.3f}], median={np.median(along):+.3f}',
        '|residual|/|change|': f'{(residual_norm / (change_norm + EPS)).mean():.2f}',
        'mean |change|': f'{change_norm.mean():.3f}',
        'mean |residual|': f'{residual_norm.mean():.3f}',
    }
    _print_section('[G2] component of the residual along the change, as a fraction of the change', items)
    return along


if __name__ == '__main__':
    main()
