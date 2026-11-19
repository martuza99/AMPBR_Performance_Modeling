"""R1.4 (round 3): influence of the two extreme BMP records (230 mg/L.d, study [20]; 158 mg/L.d, study [23]).
Both are single-record studies, so excluding the records also excludes their source studies.
(a) metric-only: reported pooled predictions re-scored without the two records (no retraining);
(b) retrained : all six models rerun with and without the two records, nested 5-fold CV (random 80/20 inner split,
    as in Table 4) and LOSO (study-grouped inner validation, as in the R1.2 check); same candidates in both subsets.
Outputs: BMP_extremes_metric_only.csv, BMP_extremes_retrained.csv, BMP_extremes_predictions.csv"""
import os, sys, time, warnings
import numpy as np, pandas as pd
from joblib import Parallel, delayed
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r3common import *

EXTREMES = ('[20]', '[23]')
MODELS = ['Ridge', 'SVR', 'DNN', 'XGB', 'GB', 'RF']
LOG = open(os.path.join(HERE, 'bmp_extremes.log'), 'a', encoding='utf-8')


def log(s):
    print(s, flush=True); LOG.write(s + '\n'); LOG.flush()


def fold(m, X, y, g, tr, te, k, scheme):
    Xtr, Xte = impute(X[tr], X[te])
    p, h = tune_predict(m, Xtr, y[tr], g[tr], Xte, cands_for(m, 'BMP', k, salt=0 if scheme == 'LOSO' else 500),
                        'grouped' if scheme == 'LOSO' else 'random')
    return te, p


def metric_only():
    rows = []
    for f, scheme in (('KFold_v2_predictions_all_models.csv', '5-fold CV'), ('LOSO_v2_predictions_all_models.csv', 'LOSO')):
        d = pd.read_csv(os.path.join(HERE, f))
        if scheme == 'LOSO':      # Ridge / RF / SVR LOSO predictions of Table 5
            d = pd.concat([d, pd.read_csv(os.path.join(HERE, 'LOSO_RidgeRFSVR_fit80_predictions.csv'))])
        d = d[d.Target == 'BMP']
        for mname, s in d.groupby('Model'):
            ext = s.Actual.round(2).isin([230.0, 158.18])
            a, b = metr(s.Actual, s.Predicted), metr(s.Actual[~ext], s.Predicted[~ext])
            rows.append(dict(Scheme=scheme, Model=mname, N_all=len(s), N_excl=int((~ext).sum()),
                             **{f'{k}_all': v for k, v in a.items()}, **{f'{k}_excl': v for k, v in b.items()}))
    out = pd.DataFrame(rows); out.to_csv(os.path.join(HERE, 'BMP_extremes_metric_only.csv'), index=False)
    log(out[['Scheme', 'Model', 'R2_all', 'R2_excl', 'MAE_all', 'MAE_excl']].round(3).to_string())


if __name__ == '__main__':
    log(f'=== run {time.ctime()}')
    metric_only()
    if 'metric' in sys.argv: sys.exit()
    rows, preds = [], []
    for scheme in ('5-fold CV', 'LOSO'):
        for m in MODELS:
            r = {}
            for subset, excl in (('all', ()), ('excl', EXTREMES)):
                X, y, g = dataset('BMP', excl)
                sp = kfold_splits(y) if scheme == '5-fold CV' else loso_splits(g)
                t0 = time.time()
                res = Parallel(n_jobs=4)(delayed(fold)(m, X, y, g, tr, te, k, scheme) for tr, te, k in sp)
                p = np.empty_like(y)
                for te, pt in res: p[te] = pt
                r[subset] = metr(y, p)
                preds.append(pd.DataFrame(dict(Scheme=scheme, Model=m, Subset=subset, Study_ID=g, Actual=y, Predicted=p)))
                log(f'{scheme:9s} {m:6s} {subset:4s} n={len(y)} R2 = {r[subset]["R2"]:7.3f} MAE = {r[subset]["MAE"]:.2f} '
                    f'({time.time() - t0:.0f} s)')
            rows.append(dict(Scheme=scheme, Model=m, **{f'{k}_{s}': v for s in r for k, v in r[s].items()}))
            pd.DataFrame(rows).to_csv(os.path.join(HERE, 'BMP_extremes_retrained.csv'), index=False)
            pd.concat(preds).to_csv(os.path.join(HERE, 'BMP_extremes_predictions.csv'), index=False)
    log('=== done ' + time.ctime())
