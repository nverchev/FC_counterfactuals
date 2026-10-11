"""Test of the assumption behind the counterfactual construction

    counterfactual = clip(x - dec(z, p) + dec(z*, p*))

namely that the reconstruction error r = x - dec(z, p) carries no
classification-relevant information, so that it can be transplanted unchanged
onto the counterfactual reconstruction.

Tests (per conditioning model, per partition):
  T1   multivariate decoding of the diagnosis label from r (5-fold CV AUC,
       two-sided label-permutation null)
  T1b  projection of r on a class-discriminative direction estimated from the
       *originals* of an independent set (1 d.o.f. -> much higher power)
  T2   decoding of the conditioning probability from r (ridge CV R^2)
  T3   dependence of the size of r on the class
  T4   functional test on the conditioning classifier itself: does r push it
       towards the subject's own class, and does a *donated* r inject the
       donor's class?
  T5   what transplanting r costs the counterfactual

Usage: python test_recon_independence.py [--cond svc mlp ae] [--n_perm 200]
"""
import argparse
import warnings

import numpy as np
import torch
from scipy import stats
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.independence_utils import get_partition, load_setup, reconstruct

warnings.filterwarnings('ignore')
EPS = 1e-6


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def make_decoder(kind):
    if kind == 'logreg':
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=3000))
    return make_pipeline(StandardScaler(), SVC(C=100, kernel='rbf'))


def cv_auc(X, y, kind='logreg', seed=0, n_splits=5):
    cv = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    scores = cross_val_predict(make_decoder(kind), X, y, cv=cv, method='decision_function')
    return roc_auc_score(y, scores)


def cv_r2(X, t, seed=0, n_splits=5):
    cv = KFold(n_splits, shuffle=True, random_state=seed)
    model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(0, 6, 13)))
    pred = cross_val_predict(model, X, t, cv=cv)
    return 1 - ((t - pred) ** 2).sum() / ((t - t.mean()) ** 2).sum()


def class_direction(X, y):
    w = X[y == 1].mean(0) - X[y == 0].mean(0)
    return w / np.linalg.norm(w)


def perm_p_two_sided(observed, null, center=0.5):
    null = np.asarray(null)
    hits = (np.abs(null - center) >= abs(observed - center)).sum()
    return (hits + 1) / (len(null) + 1)


def banner(text, char='='):
    print('\n' + char * 78 + f'\n{text}\n' + char * 78, flush=True)


def analyse(cond, exp, n_perm, n_swaps, seed):
    trainer, clf, args = load_setup(cond, exp)
    rng = np.random.default_rng(seed)
    rows = []

    data = {}
    for part in ('train', 'test'):
        X, y, p = get_partition(trainer, part)
        data[part] = (X, y, p, X - reconstruct(trainer, X, p))

    for part, part_name in [('train', 'train_val (seen by the VAE)'), ('test', 'test (held out)')]:
        X, y, p, r = data[part]
        rec, n = X - r, len(y)
        banner(f'cond={cond}  |  {part_name}  |  n={n}  |  recon MSE={np.mean(r ** 2):.5f}')

        # ------- T1: is the label multivariately decodable from r? -----------
        print('\n[T1] Decoding the diagnosis label (5-fold CV AUC)', flush=True)
        auc_r = {}
        for feat_name, feat in [('original x', X), ('reconstruction x_hat', np.clip(rec, -1, 1)),
                                ('residual r = x - x_hat', r)]:
            a = {k: cv_auc(feat, y, k, seed) for k in ('logreg', 'svc_rbf')}
            if feat_name.startswith('residual'):
                auc_r = a
            print(f'    {feat_name:<24s} logreg AUC={a["logreg"]:.3f}   rbf-SVC AUC={a["svc_rbf"]:.3f}')
        null = [cv_auc(r, rng.permutation(y), 'logreg', seed) for _ in range(n_perm)]
        p_t1 = perm_p_two_sided(auc_r['logreg'], null)
        print(f'    label-permutation null for r ({n_perm} perms): mean={np.mean(null):.3f} '
              f'sd={np.std(null):.3f}  -> two-sided p={p_t1:.3f}')
        if auc_r['logreg'] < 0.4:
            print('    NOTE: AUC far *below* chance. With p >> n and residuals whose within-class '
                  'mean\n          is near zero, a held-out sample is anti-correlated with the sum of '
                  'its own\n          class in the training fold ("anti-learning"): the fitted decoder '
                  'inverts.\n          Read T1b/T4-swap instead, which are immune to this.')

        # ------- T1b: projection on an independently estimated class axis ----
        print('\n[T1b] Projection on a class axis fitted on independent ORIGINAL data', flush=True)
        if part == 'test':
            w = class_direction(*data['train'][:2])           # from train_val originals
            proj = {name: F @ w for name, F in [('x', X), ('x_hat', rec), ('r', r)]}
            src = 'axis from train_val originals'
        else:                                                  # CV inside train_val
            cv = StratifiedKFold(5, shuffle=True, random_state=seed)
            proj = {name: np.zeros(n) for name in ('x', 'x_hat', 'r')}
            for tr, te in cv.split(X, y):
                w = class_direction(X[tr], y[tr])
                for name, F in [('x', X), ('x_hat', rec), ('r', r)]:
                    proj[name][te] = F[te] @ w
            src = '5-fold CV axis from originals'
        print(f'    ({src})')
        auc_proj = {}
        for name, s in proj.items():
            auc_proj[name] = roc_auc_score(y, s)
            u = stats.mannwhitneyu(s[y == 1], s[y == 0])
            print(f'    {name:<6s} AUC={auc_proj[name]:.3f}   Mann-Whitney p={u.pvalue:.2e}   '
                  f'mean(ASD)={s[y == 0].mean():+.3f}  mean(control)={s[y == 1].mean():+.3f}')

        # ------- T2: conditioning probability decodable from r? --------------
        print('\n[T2] Decoding the conditioning probability from r (ridge, 5-fold CV R^2)', flush=True)
        target = logit(p).astype(np.float32)
        r2_r, r2_x = cv_r2(r, target, seed), cv_r2(X, target, seed)
        null2 = [cv_r2(r, rng.permutation(target), seed) for _ in range(max(n_perm // 4, 25))]
        p_t2 = (np.sum(np.asarray(null2) >= r2_r) + 1) / (len(null2) + 1)
        print(f'    from x : R^2={r2_x: .3f}')
        print(f'    from r : R^2={r2_r: .3f}   (permutation null mean={np.mean(null2): .3f}, p={p_t2:.3f})')

        # ------- T3: size of the reconstruction error vs class ---------------
        print('\n[T3] Magnitude of the reconstruction error vs class', flush=True)
        energy = (r ** 2).mean(1)
        u, rho = stats.mannwhitneyu(energy[y == 0], energy[y == 1]), stats.spearmanr(energy, p)
        print(f'    per-sample MSE: ASD={energy[y == 0].mean():.5f}  control={energy[y == 1].mean():.5f}  '
              f'Mann-Whitney p={u.pvalue:.3f}')
        print(f'    Spearman(MSE, conditioning prob) = {rho.statistic: .3f}  (p={rho.pvalue:.3f})')

        # ------- T4: functional test on the conditioning classifier ----------
        print('\n[T4] Does r carry class signal *the conditioning classifier uses*?', flush=True)
        p_orig, p_rec = clf(X), clf(np.clip(rec, -1, 1))
        auc_orig, auc_rec = roc_auc_score(y, p_orig), roc_auc_score(y, p_rec)
        d_logit = logit(p_orig) - logit(p_rec)
        towards = d_logit * np.where(y == 1, 1.0, -1.0)   # + = r moves clf towards the true class
        print(f'    AUC(clf | x)     = {auc_orig:.3f}      AUC(clf | x_hat) = {auc_rec:.3f}')
        print(f'    logit shift from adding r, signed towards the true class: mean={towards.mean(): .3f}'
              f'  (t-test p={stats.ttest_1samp(towards, 0).pvalue:.2e})')
        swap_donor, swap_self, swap_auc = [], [], []
        for _ in range(n_swaps):
            perm = rng.permutation(n)
            while np.any(perm == np.arange(n)):
                perm = rng.permutation(n)
            d = logit(clf(np.clip(rec + r[perm], -1, 1))) - logit(p_rec)
            swap_donor.append(roc_auc_score(y[perm], d))
            swap_self.append(roc_auc_score(y, d))
            swap_auc.append(roc_auc_score(y, clf(np.clip(rec + r[perm], -1, 1))))
        print(f'    residual swap ({n_swaps} derangements):')
        print(f'        AUC(injected logit shift -> DONOR label)     = '
              f'{np.mean(swap_donor):.3f} +- {np.std(swap_donor):.3f}')
        print(f'        AUC(injected logit shift -> recipient label) = '
              f'{np.mean(swap_self):.3f} +- {np.std(swap_self):.3f}')
        print(f'        AUC(clf | x_hat + donated r) = {np.mean(swap_auc):.3f} '
              f'+- {np.std(swap_auc):.3f}   (vs {auc_rec:.3f} with no r at all)')

        # ------- T5: what it costs the counterfactual ------------------------
        print('\n[T5] Cost of transplanting r onto the counterfactual', flush=True)
        for target_name, tgt in [('opposite', np.less(p, 0.5).astype(np.float32)),
                                 ('half', np.full_like(p, 0.5))]:
            cf, _, gen_con = trainer.generate_counterfactuals(
                *[torch.from_numpy(a).float() for a in (X, p.reshape(-1, 1), tgt.reshape(-1, 1))])
            p_con, p_cf = clf(np.clip(gen_con.numpy(), -1, 1)), clf(cf.numpy())
            back = (logit(p_cf) - logit(p_con)) * np.where(y == 1, 1.0, -1.0)
            flip_con = (np.greater_equal(p_con, .5) != np.greater_equal(p, .5)).mean()
            flip_cf = (np.greater_equal(p_cf, .5) != np.greater_equal(p, .5)).mean()
            print(f'    target={target_name:<9s} mean prob {p_con.mean():.3f} (clean x_hat*) -> '
                  f'{p_cf.mean():.3f} (+r)   |   flip rate {flip_con:.1%} -> {flip_cf:.1%}   |   '
                  f'rank corr(p) = {stats.spearmanr(p_con, p_cf).statistic:.3f}')
            print(f'{"":>21s}logit pull back towards the true class from r: mean={back.mean(): .3f} '
                  f'(p={stats.ttest_1samp(back, 0).pvalue:.2e})')

        rows.append(dict(cond=cond, part=part, auc_r=auc_r['logreg'], p_t1=p_t1,
                         proj_x=auc_proj['x'], proj_r=auc_proj['r'], r2_r=r2_r,
                         auc_orig=auc_orig, auc_rec=auc_rec,
                         swap=float(np.mean(swap_donor)), pull=float(towards.mean())))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cond', nargs='+', default=['svc', 'mlp', 'ae'], choices=['svc', 'mlp', 'ae'])
    ap.add_argument('--exp', default='final')
    ap.add_argument('--n_perm', type=int, default=200)
    ap.add_argument('--n_swaps', type=int, default=20)
    ap.add_argument('--seed', type=int, default=2023)
    a = ap.parse_args()

    rows = [row for cond in a.cond for row in analyse(cond, a.exp, a.n_perm, a.n_swaps, a.seed)]

    banner('SUMMARY   H0: the reconstruction error carries no class-relevant information')
    hdr = (f'{"cond":<5s}{"partition":<11s}{"AUC(y|r)":>9s}{"p":>7s}{"projAUC r":>11s}'
           f'{"projAUC x":>11s}{"R2(p|r)":>9s}{"AUC(y|x)":>10s}{"AUC(y|xh)":>11s}{"swap":>7s}{"pull":>7s}')
    print(hdr + '\n' + '-' * len(hdr))
    for v in rows:
        print(f'{v["cond"]:<5s}{v["part"]:<11s}{v["auc_r"]:>9.3f}{v["p_t1"]:>7.3f}{v["proj_r"]:>11.3f}'
              f'{v["proj_x"]:>11.3f}{v["r2_r"]:>9.3f}{v["auc_orig"]:>10.3f}{v["auc_rec"]:>11.3f}'
              f'{v["swap"]:>7.3f}{v["pull"]:>7.2f}')
    print("""
projAUC = label AUC of the projection on a class axis fitted on independent originals
          (T1b; 0.5 under H0).  swap = AUC of the logit shift injected by a *donated*
          residual against the DONOR label (0.5 under H0).  pull = mean logit shift the
          residual gives the conditioning classifier towards the subject's true class.""")


if __name__ == '__main__':
    main()
