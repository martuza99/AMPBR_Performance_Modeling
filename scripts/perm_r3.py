"""R1.3 (round 3): permutation tests with hyperparameters re-tuned in every permutation, and a study-level null.

Permutation types
  record : the target is shuffled across records (as before);
  study  : the study means of the target are shuffled across studies and each record keeps its deviation from its
           own study mean: y* = y - mean(study) + mean(pi(study)). Within-study structure and the set of study-level
           targets are preserved; only the association between a study's conditions and its target level is broken.
Procedure: the complete outer validation procedure, INCLUDING hyperparameter selection, is repeated for every
  permuted target (same candidates as for the real target):
  LOSO   : study-grouped inner validation (as in the R1.2 check); Ridge and SVR, every target with positive LOSO R2;
           record- and study-level permutation, 99 each;
  5-fold : the reported folds (fold_assignment_*.csv), random 80/20 inner split (as in Table 4); Ridge, SVR, DNN and
           XGB, all targets; record-level permutation (99 for Ridge/SVR, 49 for DNN/XGB).
  p = (1 + #null R2 >= real R2) / (1 + n_perm), the real R2 being obtained with the same pipeline.
Output: Permutation_r3_retuned.csv, Permutation_r3_null.csv"""
import os, sys, time, warnings
import numpy as np, pandas as pd
from joblib import Parallel, delayed
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r3common import *

LOG = open(os.path.join(HERE, 'perm_r3.log'), 'a', encoding='utf-8')
TEST = [a for a in sys.argv[1:] if a.isdigit()]


def log(s):
    print(s, flush=True); LOG.write(s + '\n'); LOG.flush()


def study_perm(y, g, rng):
    st = np.unique(g)
    mu = {s: y[g == s].mean() for s in st}
    pi = dict(zip(st, rng.permutation(st)))
    return np.array([v - mu[s] + mu[pi[s]] for v, s in zip(y, g)])


def outer_r2(m, X, y, g, splits, scheme, T):
    p = np.empty_like(y)
    for tr, te, k in splits:
        Xtr, Xte = impute(X[tr], X[te])
        p[te] = tune_predict(m, Xtr, y[tr], g[tr], Xte, cands_for(m, T, k, salt=0 if scheme == 'LOSO' else 500),
                             'grouped' if scheme == 'LOSO' else 'random')[0]
    return r2_score(y, p)


def user_kfold(T, n):
    fa = pd.read_csv(os.path.join(HERE, f'fold_assignment_{T}.csv')).sort_values('row_in_target_subset')
    assert len(fa) == n
    f = fa.fold.to_numpy()
    return [(np.where(f != k)[0], np.where(f == k)[0], int(k)) for k in sorted(np.unique(f))]


if __name__ == '__main__':
    gm = pd.read_csv(os.path.join(HERE, 'LOSO_grouped_inner_metrics.csv'))
    cases = []
    for T in ['BMP', 'COD', 'NR', 'PR']:
        for m in ['Ridge', 'SVR']:
            r = gm[(gm.Target == T) & (gm.Model == m)]
            if len(r) and r.R2_grouped.iloc[0] > 0:
                cases += [('LOSO', T, m, 'record', 99), ('LOSO', T, m, 'study', 99)]
    for T in ['BMP', 'COD', 'NR', 'PR']:
        for m, n in (('Ridge', 99), ('SVR', 99), ('DNN', 49), ('XGB', 49)):
            cases.append(('5-fold CV', T, m, 'record', n))
    if TEST: cases = cases[:1]
    f_rows, f_null = os.path.join(HERE, 'Permutation_r3_retuned.csv'), os.path.join(HERE, 'Permutation_r3_null.csv')
    rows, nulls = [], []
    if os.path.exists(f_rows) and not TEST:
        rows, nulls = pd.read_csv(f_rows).to_dict('records'), pd.read_csv(f_null).to_dict('records')
    done = {(r['Scheme'], r['Target'], r['Model'], r['Perm_type']) for r in rows}
    log(f'=== run {time.ctime()}  {len(cases)} cases')
    for ci, (scheme, T, m, ptype, n) in enumerate(cases):
        if (scheme, T, m, ptype) in done: continue
        if TEST: n = int(TEST[0])
        X, y, g = dataset(T)
        splits = loso_splits(g) if scheme == 'LOSO' else user_kfold(T, len(y))
        t0 = time.time()
        real = outer_r2(m, X, y, g, splits, scheme, T)
        rng = np.random.default_rng(7000 + ci)
        yps = [rng.permutation(y) if ptype == 'record' else study_perm(y, g, rng) for _ in range(n)]
        null = np.array(Parallel(n_jobs=4)(delayed(outer_r2)(m, X, yp, g, splits, scheme, T) for yp in yps))
        rows.append(dict(Scheme=scheme, Target=T, Model=m, Perm_type=ptype, Retuned=True, R2_real=real, n_perm=n,
                         null_mean=null.mean(), null_95pct=np.percentile(null, 95), null_max=null.max(),
                         p_value=(1 + (null >= real).sum()) / (1 + n)))
        nulls += [dict(Scheme=scheme, Target=T, Model=m, Perm_type=ptype, perm=i, R2_null=v) for i, v in enumerate(null)]
        log(f'{scheme:9s} {T:4s} {m:6s} {ptype:6s} real {real:6.3f} | null mean {null.mean():6.2f} 95% '
            f'{np.percentile(null, 95):6.2f} max {null.max():6.2f} | p = {rows[-1]["p_value"]:.3f} ({time.time() - t0:.0f} s)')
        if not TEST:
            pd.DataFrame(rows).to_csv(f_rows, index=False); pd.DataFrame(nulls).to_csv(f_null, index=False)
    log('=== done ' + time.ctime())
