"""EDS-only vs CSI-augmented training under the authors' tuning procedure (R5.2), both schemes, Ridge/SVR/RF/GB/XGB.
Per outer fold (user's fold files, imputed per fold):
  fit, val = train_test_split(training records, 0.2, random_state=42)       (same split as the EDS-only runs)
  CSI-augmented fitting set = fit + 500 records interpolated from fit only (manuscript CSI: sort by target, natural cubic
      splines of every column against the rank index, clipping to the fit range, physical-constraint filter)
  StandardScaler on the augmented fitting set; bayes_opt tuning on the ORIGINAL validation records (R2);
  final model kept as fitted on the augmented fitting set; held-out original records predicted.
EDS-only results are the existing tuned runs (same folds, split and procedure without augmentation).
Resumable: one CSV of predictions per (scheme, target, model)."""
import os, sys, time, warnings
import numpy as np, pandas as pd
from scipy.interpolate import CubicSpline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
from bayes_opt import BayesianOptimization
from joblib import Parallel, delayed

warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from round2_bo_loso import SPEC                                     # Ridge / SVR / RF exactly as in the user's notebooks

DIRS = {'KF': r'E:\My Drive\Colab Notebooks\AMBR Prediction_v5kcv2', 'LOSO': r'E:\My Drive\Colab Notebooks\AMBR Prediction_v5loso'}
OUTD = os.path.join(HERE, 'csi_tuned'); os.makedirs(OUTD, exist_ok=True)
LOG = open(os.path.join(HERE, 'csi_compare_tuned.log'), 'a', encoding='utf-8')
N_CSI, N_JOBS = 500, 4
INF = {'COD': 0, 'NR': 1, 'PR': 2}                                   # influent column of each effluent target


def log(s):
    print(s, flush=True); LOG.write(s + '\n'); LOG.flush()


def csi(fit, T):
    a = fit.to_numpy(float); y = a[:, -1]
    o = np.argsort(y, kind='stable'); s = a[o]; t = np.arange(len(s), dtype=float)
    tq = np.linspace(0, len(s) - 1, N_CSI)
    q = np.column_stack([CubicSpline(t, s[:, j], bc_type='natural')(tq) for j in range(a.shape[1])])
    q = np.clip(q, a.min(0), a.max(0))
    ok = np.ones(len(q), bool)
    if T in INF:
        ok &= q[:, -1] <= q[:, INF[T]]                               # effluent <= influent
    ok &= q[:, 7] >= q[:, 6]                                         # SRT >= HRT
    return pd.DataFrame(np.vstack([a, q[ok]]), columns=fit.columns), int(ok.sum())


def two_stage(kind, Xf, yf, Xv, yv):
    if kind == 'XGB':
        ip, ni = 20, 200
        mk = lambda p: xgb.XGBRegressor(objective='reg:squarederror', n_jobs=1, **p)
        b1 = {'n_estimators': (100, 1500), 'learning_rate': (1e-5, 1)}
        f1 = lambda n_estimators, learning_rate: dict(n_estimators=int(n_estimators), learning_rate=learning_rate)
        b2 = {'max_depth': (1, 10), 'min_child_weight': (1, 10), 'subsample': (0.5, 1.0), 'colsample_bytree': (0.5, 1.0)}
        f2 = lambda base, max_depth, min_child_weight, subsample, colsample_bytree: dict(
            base, max_depth=int(max_depth), min_child_weight=int(min_child_weight), subsample=subsample, colsample_bytree=colsample_bytree)
    else:
        ip, ni = 10, 100
        mk = lambda p: GradientBoostingRegressor(random_state=1, **p)
        b1 = {'n_estimators': (100, 1500), 'learning_rate': (1e-3, 0.5), 'max_depth': (1, 10)}
        f1 = lambda n_estimators, learning_rate, max_depth: dict(n_estimators=int(round(n_estimators)), learning_rate=learning_rate,
                                                                 max_depth=int(round(max_depth)))
        b2 = {'min_samples_split': (2, 20), 'min_samples_leaf': (1, 10), 'subsample': (0.5, 1.0), 'max_features': (0.3, 1.0)}
        f2 = lambda base, min_samples_split, min_samples_leaf, subsample, max_features: dict(
            base, min_samples_split=int(round(min_samples_split)), min_samples_leaf=int(round(min_samples_leaf)),
            subsample=subsample, max_features=max_features)
    o1 = BayesianOptimization(f=lambda **p: r2_score(yv, mk(f1(**p)).fit(Xf, yf).predict(Xv)), pbounds=b1, random_state=1, verbose=0)
    o1.maximize(init_points=ip, n_iter=ni); base = f1(**o1.max['params'])
    o2 = BayesianOptimization(f=lambda **p: r2_score(yv, mk(f2(base, **p)).fit(Xf, yf).predict(Xv)), pbounds=b2, random_state=1, verbose=0)
    o2.maximize(init_points=ip, n_iter=ni)
    return mk(f2(base, **o2.max['params'])).fit(Xf, yf)


def one_fold(scheme, T, k, M):
    D = DIRS[scheme]
    tr = pd.read_csv(os.path.join(D, f'Training_data_ori_{T}-fold{k}.csv'), encoding='latin-1')
    te = pd.read_csv(os.path.join(D, f'Testing_data_ori_{T}-fold{k}.csv'), encoding='latin-1')
    fit, val = train_test_split(tr, test_size=0.20, random_state=42, shuffle=True)
    aug, nadd = csi(fit, T)
    sc = StandardScaler().fit(aug)
    d1, d2, d3 = sc.transform(aug), sc.transform(te), sc.transform(val)
    Xf, yf, Xt, Xv, yv = d1[:, :-1], d1[:, -1], d2[:, :-1], d3[:, :-1], d3[:, -1]
    if M in SPEC:
        sp = SPEC[M]
        opt = BayesianOptimization(f=lambda **p: r2_score(yv, sp['make'](p, 1).fit(Xf, yf).predict(Xv)), pbounds=sp['pb'],
                                   random_state=1, verbose=0)
        opt.maximize(init_points=sp['initp'], n_iter=sp['niter'])
        m = sp['make'](opt.max['params'], 1).fit(Xf, yf)
    else:
        m = two_stage(M, Xf, yf, Xv, yv)
    return pd.DataFrame(dict(Scheme=scheme, Target=T, Model=M, Fold=k, Actual=te.iloc[:, -1].to_numpy(float),
                             Predicted=m.predict(Xt) * sc.scale_[-1] + sc.mean_[-1], N_fit=len(fit), N_csi_added=nadd))


if __name__ == '__main__':
    models = sys.argv[1].split(',') if len(sys.argv) > 1 else ['Ridge', 'SVR', 'RF', 'XGB', 'GB']
    fmap = pd.read_csv(os.path.join(DIRS['LOSO'], 'LOSO_fold_to_study.csv'))
    log(f'=== run {time.ctime()} models {models}')
    for M in models:
        for scheme in ['KF', 'LOSO']:
            for T in ['BMP', 'COD', 'NR', 'PR']:
                f = os.path.join(OUTD, f'{scheme}_{T}_{M}.csv')
                if os.path.exists(f):
                    continue
                folds = range(1, 6) if scheme == 'KF' else sorted(fmap[fmap.target == T].fold.astype(int))
                t0 = time.time()
                res = pd.concat(Parallel(n_jobs=N_JOBS)(delayed(one_fold)(scheme, T, k, M) for k in folds), ignore_index=True)
                res.to_csv(f, index=False)
                log(f'{scheme:4s} {T:4s} {M:5s} CSI-augmented R2 = {r2_score(res.Actual, res.Predicted):7.3f} '
                    f'(+{int(res.groupby("Fold").N_csi_added.first().mean())} interpolated/fold; {time.time() - t0:.0f} s)')
    log('=== done ' + time.ctime())
