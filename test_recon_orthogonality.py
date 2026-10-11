"""Tests the *weaker* assumption actually required by the counterfactual
construction  cf = clip(x + d),  d = dec(z*,p*) - dec(z,p):

    the reconstruction error r = x - dec(z,p) is uncorrelated with the
    counterfactual change d.

r is allowed to carry class information; what it must not do is carry it along
the direction the intervention moves in, because then transplanting r
attenuates (or inflates) the very effect the metrics measure.

  G1  geometry: cos(r, d) against a derangement null (r donated by another
      subject), plus the class dependence of that alignment
  G2  scale: the component of r along d, expressed as a fraction of ||d||
      (-1 = the residual cancels the whole intervention)
  G3  function: is the classifier's response to d the same with r, without r,
      and with a donated r?  (no-interaction form of the same assumption)

Usage: python test_recon_orthogonality.py [--cond svc mlp ae] [--n_null 200]
"""
import argparse
import warnings

import numpy as np
import torch
from scipy import stats

from src.independence_utils import get_partition, load_setup, reconstruct

warnings.filterwarnings('ignore')
EPS = 1e-6


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def cosine(a, b):
    return np.sum(a * b, 1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + EPS)


def derangement(n, rng):
    perm = rng.permutation(n)
    while np.any(perm == np.arange(n)):
        perm = rng.permutation(n)
    return perm


def banner(text, char='='):
    print('\n' + char * 78 + f'\n{text}\n' + char * 78, flush=True)


@torch.inference_mode()
def decode_pair(trainer, X, p, target):
    """Returns (x_hat_rec, x_hat_con) exactly as generate_counterfactuals does."""
    _, rec, con = trainer.generate_counterfactuals(
        *[torch.from_numpy(a).float() for a in (X, p.reshape(-1, 1), target.reshape(-1, 1))])
    return rec.numpy(), con.numpy()


def analyse(cond, exp, n_null, seed):
    trainer, clf, args = load_setup(cond, exp)
    rng = np.random.default_rng(seed)
    rows = []

    for part, part_name in [('train', 'train_val (seen by the VAE)'), ('test', 'test (held out)')]:
        X, y, p = get_partition(trainer, part)
        n = len(y)
        for target_name, target in [('opposite', np.less(p, 0.5).astype(np.float32)),
                                    ('half', np.full_like(p, 0.5))]:
            rec, con = decode_pair(trainer, X, p, target)
            r = X - rec
            d = con - rec                      # the counterfactual change
            banner(f'cond={cond} | {part_name} | target={target_name} | n={n}')

            # ---------------- G1: alignment of r with d ----------------------
            cos = cosine(r, d)
            null = np.concatenate([cosine(r[derangement(n, rng)], d) for _ in range(max(n_null // 10, 10))])
            t = stats.ttest_1samp(cos, null.mean())
            print(f'\n[G1] cos(r, d): mean={cos.mean():+.4f}  sd={cos.std():.4f}')
            print(f'     donated-residual null: mean={null.mean():+.4f}  sd={null.std():.4f}')
            print(f'     observed vs null mean: t-test p={t.pvalue:.2e}   '
                  f'(Cohen d={(cos.mean() - null.mean()) / cos.std():+.2f})')
            u = stats.mannwhitneyu(cos[y == 0], cos[y == 1])
            print(f'     by class: ASD={cos[y == 0].mean():+.4f}  control={cos[y == 1].mean():+.4f}  '
                  f'Mann-Whitney p={u.pvalue:.2e}')
            pb = stats.pointbiserialr(y, cos)
            print(f'     point-biserial corr(label, cos) = {pb.statistic:+.3f} (p={pb.pvalue:.2e})')

            # ---------------- G2: how much of d does r undo? -----------------
            dn = np.linalg.norm(d, axis=1)
            frac = np.sum(r * d, 1) / (dn ** 2 + EPS)      # r's component along d, in units of d
            ci = stats.t.interval(0.95, n - 1, loc=frac.mean(), scale=stats.sem(frac))
            print(f'\n[G2] component of r along d, as a fraction of d')
            print(f'     mean={frac.mean():+.3f}  95% CI=[{ci[0]:+.3f}, {ci[1]:+.3f}]  '
                  f'median={np.median(frac):+.3f}')
            print(f'     ||r||/||d|| = {(np.linalg.norm(r, axis=1) / (dn + EPS)).mean():.2f}   '
                  f'(mean ||d||={dn.mean():.3f}, mean ||r||={np.linalg.norm(r, axis=1).mean():.3f})')

            # ---------------- G3: no-interaction test on the classifier ------
            perm = derangement(n, rng)
            base_rec, base_x = clf(np.clip(rec, -1, 1)), clf(X)
            eff_without = logit(clf(np.clip(con, -1, 1))) - logit(base_rec)
            eff_with = logit(clf(np.clip(X + d, -1, 1))) - logit(base_x)
            eff_donated = (logit(clf(np.clip(rec + r[perm] + d, -1, 1)))
                           - logit(clf(np.clip(rec + r[perm], -1, 1))))
            clipped = np.mean(np.abs(X + d) > 1)
            print(f'\n[G3] classifier response to the same d ({clipped:.2%} of entries clipped)')
            print(f'     effect without r (on x_hat)  : mean={eff_without.mean():+.3f}  '
                  f'sd={eff_without.std():.3f}')
            print(f'     effect with r    (actual cf) : mean={eff_with.mean():+.3f}  '
                  f'sd={eff_with.std():.3f}')
            print(f'     effect with a donated r      : mean={eff_donated.mean():+.3f}  '
                  f'sd={eff_donated.std():.3f}')
            att = eff_with.mean() / eff_without.mean() if eff_without.mean() else np.nan
            print(f'     attenuation with/without = {att:+.3f}   '
                  f'corr(with, without) = {np.corrcoef(eff_with, eff_without)[0, 1]:+.3f}')
            dif = stats.ttest_rel(eff_with, eff_without)
            print(f'     paired difference (with - without): mean='
                  f'{(eff_with - eff_without).mean():+.3f}  p={dif.pvalue:.2e}')
            # does the interaction track r's alignment with d?
            inter = eff_with - eff_without
            sp = stats.spearmanr(frac, inter)
            print(f'     Spearman(r-along-d, interaction) = {sp.statistic:+.3f} (p={sp.pvalue:.2e})')

            rows.append(dict(cond=cond, part=part, target=target_name, cos=cos.mean(),
                             null=null.mean(), frac=frac.mean(), att=att,
                             corr=np.corrcoef(eff_with, eff_without)[0, 1],
                             pb=pb.statistic))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cond', nargs='+', default=['svc', 'mlp', 'ae'], choices=['svc', 'mlp', 'ae'])
    ap.add_argument('--exp', default='final')
    ap.add_argument('--n_null', type=int, default=200)
    ap.add_argument('--seed', type=int, default=2023)
    a = ap.parse_args()

    rows = [row for cond in a.cond for row in analyse(cond, a.exp, a.n_null, a.seed)]

    banner('SUMMARY   H0: the residual is uncorrelated with the counterfactual change')
    hdr = (f'{"cond":<5s}{"partition":<11s}{"target":<10s}{"cos(r,d)":>10s}{"null":>9s}'
           f'{"r along d":>11s}{"attenuat.":>11s}{"corr w/wo":>11s}{"corr(y,cos)":>13s}')
    print(hdr + '\n' + '-' * len(hdr))
    for v in rows:
        print(f'{v["cond"]:<5s}{v["part"]:<11s}{v["target"]:<10s}{v["cos"]:>+10.4f}{v["null"]:>+9.4f}'
              f'{v["frac"]:>+11.3f}{v["att"]:>+11.3f}{v["corr"]:>+11.3f}{v["pb"]:>+13.3f}')
    print("""
cos(r,d)    alignment of the residual with the intervention (0 under H0; compare to null)
r along d   residual's component along d in units of d (0 under H0; -1 = cancels d entirely)
attenuat.   mean classifier effect of d with the residual / without it (1 under H0)
corr w/wo   per-subject correlation of those two effects (1 under H0)
corr(y,cos) does the alignment depend on the subject's class? (0 under H0)""")


if __name__ == '__main__':
    main()
