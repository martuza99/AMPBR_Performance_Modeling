"""Shared pieces for the round-3 analyses (R1.2-R1.4).
Data: raw records (missing inputs kept) with the user's per-target targets; inputs are mean-imputed with the
training records of each outer fold, which reproduces the user's per-fold LOSO files exactly (checked)."""
import os, sys
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from loso_grouped_inner import sample, make, fit_predict, N_CAND, N_INNER   # same spaces / models as R1.2

B5 = r'E:\My Drive\Colab Notebooks\AMBR Prediction_v5loso'
RAW = r'E:\My Drive\Colab Notebooks\AMBR Prediction_v4\Data-AMBR_ori_v3.csv'
INPUTS = ['COD (mg/L)', 'N (mg/L)', 'P (mg/L)', 'pH', 'Avg. Temperature 0C', 'Flux (L/m2/h)',
          'HRT (d)', 'SRT (d)', 'Light intensity (umol/m2s)', 'Light cycle (h)']
TCOL = {'BMP': 'Productivity (mg/L·d)', 'COD': 'Effluent COD  (mg/L)', 'NR': 'Effluent N  (mg/L)', 'PR': 'Effluent P  (mg/L)'}
TIDX = {'BMP': 0, 'COD': 1, 'NR': 2, 'PR': 3}


def dataset(T, exclude_refs=()):
    d = pd.read_csv(RAW)
    d['Ref'] = pd.read_excel(os.path.join(HERE, 'Table_S1_v3.xlsx'))['Ref'].astype(str).str.strip().values
    d['Avg. Temperature 0C'] = d['Avg. Temperature 0C'].fillna(23.0)
    s = d[d[TCOL[T]].notna()].reset_index(drop=True)
    pp = pd.read_excel(os.path.join(B5, f'Table_S1_v3_pp_num_{T}.xlsx'))
    assert (pp.Ref.astype(str).str.strip().values == s.Ref.values).all()
    y = pp.iloc[:, 11].to_numpy(float)                 # the user's targets (incl. the COD harmonisation)
    keep = ~s.Ref.isin(list(exclude_refs)).values
    return s[INPUTS].to_numpy(float)[keep], y[keep], s.Ref.values[keep]


def impute(Xtr, *Xo):
    mu = np.nanmean(Xtr, 0)
    f = lambda A: np.where(np.isnan(A), mu, A)
    return (f(Xtr),) + tuple(f(A) for A in Xo)


def tune_predict(m, X, y, g, Xt, cands, scheme):
    """Select among cands and predict Xt. scheme 'random': one random 80/20 split, kept as fitted on 80%;
    'grouped': GroupKFold over the training studies, refitted on all training records."""
    if scheme == 'random':
        fi, vi = train_test_split(np.arange(len(y)), test_size=0.2, random_state=42, shuffle=True)
        sc = [r2_score(y[vi], fit_predict(m, h, X[fi], y[fi], [X[vi]])[0]) for h in cands]
        h = cands[int(np.nanargmax(sc))]
        return fit_predict(m, h, X[fi], y[fi], [Xt])[0], h
    inner = list(GroupKFold(n_splits=min(N_INNER, len(np.unique(g)))).split(X, y, g))
    sc = []
    for h in cands:
        p = np.empty_like(y)
        for a, b in inner:
            p[b] = fit_predict(m, h, X[a], y[a], [X[b]])[0]
        sc.append(r2_score(y, p))
    h = cands[int(np.nanargmax(sc))]
    return fit_predict(m, h, X, y, [Xt])[0], h


def loso_splits(g):
    studies = sorted(np.unique(g), key=lambda r: int(r.strip('[]')))
    return [(np.where(g != s)[0], np.where(g == s)[0], i + 1) for i, s in enumerate(studies)]   # fold no. as in R1.2


def kfold_splits(y, n_bins=10, n_folds=5):
    """Stratified 5-fold as in Section 2.8.1: 10 bins of ascending target, records allocated to the folds in turn."""
    order = np.argsort(y, kind='stable')
    f = np.empty(len(y), int); c = 0
    for b in np.array_split(order, n_bins):
        for i in b:
            f[i] = c % n_folds; c += 1
    return [(np.where(f != k)[0], np.where(f == k)[0], k + 1) for k in range(n_folds)]


def cands_for(m, T, k, salt=0):
    rng = np.random.default_rng(1000 * TIDX[T] + (k if isinstance(k, int) else int(str(k).strip('[]'))) + salt)
    return [sample(m, rng) for _ in range(N_CAND)]


def metr(y, p):
    return dict(R2=r2_score(y, p), MAE=mean_absolute_error(y, p), RMSE=mean_squared_error(y, p) ** 0.5)
