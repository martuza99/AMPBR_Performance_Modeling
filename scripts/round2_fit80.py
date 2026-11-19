"""
Round-2 revision runs (DWT-D-26-01475).

Part A - LOSO for Ridge, RF and SVR with nested tuning:
    outer loop  : leave-one-study-out (22/14/28/28 folds for BMP/COD/N/P);
    tuning      : as in the user's runs - one random 80/20 split of the training records (training studies only)
                  into fit / validation, RandomizedSearchCV, 20 candidates, validation MSE; the best candidate is
                  kept as fitted on the 80% fitting subset (NO refit), matching the user's XGB/GB runs;
    preprocessing (inside the pipeline, fitted on training studies only): mean imputation, standard scaling
                  of inputs and target. Unreported temperature = 23 degC (applied before any split).
Part B - permutation (null) tests, whole outer procedure repeated with the target shuffled across records:
    5-fold CV : the user's 5 folds (fold_assignment_*.csv), all six models, all targets;
    LOSO      : every model x target with positive LOSO R2 (user's runs and Part A).
    Each outer fold uses the hyperparameters selected in that fold (recovered from the saved models,
    Hyperparameters_per_fold.csv; Part A for LOSO Ridge/RF/SVR), without re-tuning. The same fixed-hyperparameter
    procedure is applied to the real target ("replica R2") and to every permuted target, so p compares like
    with like: p = (1 + #null R2 >= replica R2) / (1 + n_perm).
    DNN: replicated with scikit-learn MLPRegressor (most frequent selected architecture, ReLU, Adam, lr 0.01;
    early stopping on a 20% validation split in 5-fold CV, none in LOSO, mirroring the Keras runs).
Outputs (this folder): LOSO_RidgeRFSVR_fit80_predictions.csv, LOSO_RidgeRFSVR_fit80_metrics.csv,
    LOSO_RidgeRFSVR_fit80_hyperparameters.csv, Permutation_tests_fit80.csv, Permutation_null_distributions_fit80.csv
"""
import os, sys, time, warnings
import numpy as np, pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.model_selection import LeaveOneGroupOut, ShuffleSplit, RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from joblib import Parallel, delayed

warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import loso_validation as L

RUN = os.path.expandvars(r'%TEMP%\rev2run\v3')
L.DATA, L.TABLE_S1 = os.path.join(RUN, 'Data-AMBR_ori_v3.csv'), os.path.join(RUN, 'Table_S1_v3.xlsx')
SEED, N_ITER, N_JOBS = 0, 20, 4
N_PERM = {'Ridge': 199, 'SVR': 199, 'RF': 99, 'GB': 99, 'XGB': 99, 'DNN': 99}
RESUME_B = 'resumeB' in sys.argv            # rerun Part A, keep finished Part B cases
RESUME = 'resume' in sys.argv                # python script.py resume  -> reuse Part A and finished Part B cases
_num = [a for a in sys.argv[1:] if a.isdigit()]
if _num:                                     # quick test: python script.py 5
    N_PERM = {k: int(_num[0]) for k in N_PERM}
TKEY = {'BMP': 'BMP', 'COD_Efflu': 'COD', 'N_Efflu': 'NR', 'P_Efflu': 'PR'}
LOG = open(os.path.join(HERE, 'round2_fit80.log'), 'a', encoding='utf-8')


def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); LOG.write(s + '\n'); LOG.flush()


def data():
    d = L.load()
    c = L.TARGETS['COD_Efflu']
    d.loc[(d.Study_ID == 29) & (d[c].round(3) == 0.021), c] = 0.015688   # harmonise with the user's files
    out = {}
    for tname, tcol in L.TARGETS.items():
        s = d[d[tcol].notna()].reset_index(drop=True)
        out[TKEY[tname]] = (s[L.INPUTS].to_numpy(float), s[tcol].to_numpy(float), s.Study_ID.to_numpy())
    return out


def pipe(model):
    return TransformedTargetRegressor(make_pipeline(SimpleImputer(strategy='mean'), StandardScaler(), model),
                                      transformer=StandardScaler())


def metrics(y, p):
    return dict(R2=r2_score(y, p), MAE=mean_absolute_error(y, p), RMSE=mean_squared_error(y, p) ** 0.5)


# ------------------------------------------------------------------ Part A: tuned LOSO Ridge / RF / SVR
SPACES = {
    'Ridge': (Ridge(), {'alpha': loguniform(1e-4, 1e4)}),
    'RF': (RandomForestRegressor(random_state=SEED, n_jobs=1),
           {'n_estimators': randint(100, 1001), 'max_depth': list(range(2, 31)),
            'min_samples_split': randint(2, 21), 'min_samples_leaf': randint(1, 11), 'max_features': uniform(0.3, 0.7)}),
    'SVR': (SVR(kernel='rbf'), {'C': loguniform(1e-2, 1e3), 'epsilon': loguniform(1e-3, 1), 'gamma': loguniform(1e-4, 10)}),
}


def tuned_fold(X, y, g, tr, te, mname):
    model, grid = SPACES[mname]
    grid = {f'regressor__{type(model).__name__.lower()}__{k}': v for k, v in grid.items()}
    # same scheme as the user's runs: one random 80/20 split of the training records into fit / validation;
    # the best candidate (validation MSE) is kept as fitted on the 80% fitting subset (no refit)
    s = RandomizedSearchCV(pipe(model), grid, n_iter=N_ITER, cv=ShuffleSplit(1, test_size=0.2, random_state=SEED),
                           scoring='neg_mean_squared_error', random_state=SEED, n_jobs=1, refit=False).fit(X[tr], y[tr])
    fi = tr[fit_idx(len(tr))]
    est = pipe(model).set_params(**s.best_params_).fit(X[fi], y[fi])
    hp = {kk.split('__')[-1]: v for kk, v in s.best_params_.items()}
    return te, est.predict(X[te]), hp, fi, est.predict(X[fi])


def fit_idx(n):
    """Indices (within the training fold) of the 80% fitting subset - identical to the split used for tuning."""
    return next(ShuffleSplit(1, test_size=0.2, random_state=SEED).split(np.zeros(n)))[0]


def part_a(D):
    preds, mets, hps, trp = [], [], [], []
    for t, (X, y, g) in D.items():
        for mname in SPACES:
            t0 = time.time()
            folds = list(LeaveOneGroupOut().split(X, y, g))
            res = Parallel(n_jobs=N_JOBS)(delayed(tuned_fold)(X, y, g, tr, te, mname) for tr, te in folds)
            p = np.empty_like(y)
            ta, tp = [], []
            for i, (te, pt, hp, fi, pf) in enumerate(res, 1):
                p[te] = pt
                hps += [dict(Target=t, Model=mname, Fold=i, Study_ID=g[te][0], Param=k, Value=v) for k, v in hp.items()]
                trp.append(pd.DataFrame(dict(Target=t, Model=mname, Fold=i, Study_out=g[te][0], Study_ID=g[fi],
                                             Actual=y[fi], Predicted=pf)))
                ta.append(y[fi]); tp.append(pf)
            ta, tp = np.concatenate(ta), np.concatenate(tp)
            preds.append(pd.DataFrame(dict(Target=t, Model=mname, Study_ID=g, Actual=y, Predicted=p)))
            mets.append(dict(Target=t, Model=mname, N=len(y), Studies=len(folds), **metrics(y, p),
                             **{f'Train_{k}': v for k, v in metrics(ta, tp).items()}, N_train_rows=len(ta)))
            log(f'[A] LOSO {t:4s} {mname:6s} R2 = {mets[-1]["R2"]:7.3f}  MAE = {mets[-1]["MAE"]:.3f}  ({time.time() - t0:.0f} s)')
    pd.concat(preds).to_csv(os.path.join(HERE, 'LOSO_RidgeRFSVR_fit80_predictions.csv'), index=False)
    pd.DataFrame(mets).to_csv(os.path.join(HERE, 'LOSO_RidgeRFSVR_fit80_metrics.csv'), index=False)
    pd.DataFrame(hps).to_csv(os.path.join(HERE, 'LOSO_RidgeRFSVR_fit80_hyperparameters.csv'), index=False)
    pd.concat(trp).to_csv(os.path.join(HERE, 'LOSO_RidgeRFSVR_fit80_training_predictions.csv'), index=False)
    return pd.DataFrame(mets), pd.DataFrame(hps)


# ------------------------------------------------------------------ Part B: permutation tests
INT_P = {'n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'min_child_weight'}


def median_hp(hp, scheme, model, target):
    s = hp[(hp.Scheme == scheme) & (hp.Model == model) & (hp.Target == target)]
    out = {}
    for p, v in s.groupby('Param').Value:
        v = v.dropna()
        if len(v) == 0:
            continue
        num = pd.to_numeric(v, errors='coerce')
        if num.notna().all():
            m = float(num.median())
            out[p] = int(round(m)) if p in INT_P else m
        else:
            out[p] = v.astype(str).value_counts().index[0]
    return out


def make_model(model, hp, scheme):
    if model == 'Ridge':
        return Ridge(alpha=hp['alpha'])
    if model == 'SVR':
        return SVR(kernel='rbf', C=hp['C'], epsilon=hp['epsilon'], gamma=hp['gamma'])
    if model == 'RF':
        md = hp.get('max_depth')
        return RandomForestRegressor(n_estimators=hp['n_estimators'], max_depth=None if pd.isna(md) else int(md),
                                     min_samples_split=hp['min_samples_split'], min_samples_leaf=hp['min_samples_leaf'],
                                     max_features=float(hp['max_features']), random_state=SEED, n_jobs=1)
    if model == 'GB':
        return GradientBoostingRegressor(n_estimators=hp['n_estimators'], learning_rate=hp['learning_rate'],
                                         max_depth=hp['max_depth'], min_samples_split=hp['min_samples_split'],
                                         min_samples_leaf=hp['min_samples_leaf'], subsample=hp['subsample'],
                                         max_features=float(hp['max_features']), random_state=SEED)
    if model == 'XGB':
        return XGBRegressor(n_estimators=hp['n_estimators'], learning_rate=hp['learning_rate'], max_depth=hp['max_depth'],
                            min_child_weight=hp['min_child_weight'], subsample=hp['subsample'],
                            colsample_bytree=hp['colsample_bytree'], random_state=SEED, n_jobs=1, verbosity=0)
    if model == 'DNN':
        arch = tuple(int(u) for u in str(hp['units']).split('-'))
        return MLPRegressor(hidden_layer_sizes=arch, activation='relu', solver='adam', learning_rate_init=0.01,
                            max_iter=300, early_stopping=(scheme == '5-fold CV'), validation_fraction=0.2,
                            random_state=SEED)
    raise ValueError(model)


def fold_hps(hp, scheme, model, target, n_folds):
    """Hyperparameters selected in each outer fold (fold i = i-th split; LOSO folds are in ascending study order)."""
    s = hp[(hp.Scheme == scheme) & (hp.Model == model) & (hp.Target == target)]
    out = []
    for k in range(1, n_folds + 1):
        f = s[s.Fold == k]
        h = median_hp(f, scheme, model, target) if len(f) else median_hp(hp, scheme, model, target)
        out.append(h)
    return out


def outer_pred(X, y, splits, model, hp, scheme):
    """hp: one dict (same for every fold) or a list with one dict per fold."""
    p = np.empty_like(y)
    for i, (tr, te) in enumerate(splits):
        h = hp[i] if isinstance(hp, list) else hp
        # all models except the DNN are fitted on the 80% fitting subset (as in the user's runs);
        # the DNN uses its own 20% early-stopping split in 5-fold CV and all training records in LOSO
        fi = tr if model == 'DNN' else tr[fit_idx(len(tr))]
        p[te] = pipe(make_model(model, h, scheme)).fit(X[fi], y[fi]).predict(X[te])
    return r2_score(y, p)


def kfold_splits(t, n):
    fa = pd.read_csv(os.path.join(HERE, f'fold_assignment_{t}.csv')).sort_values('row_in_target_subset')
    assert len(fa) == n
    f = fa.fold.to_numpy()
    return [(np.where(f != k)[0], np.where(f == k)[0]) for k in sorted(np.unique(f))]


def part_b(D, loso_a):
    hp_user = pd.read_csv(os.path.join(HERE, 'Hyperparameters_per_fold.csv'))
    hp_a = pd.read_csv(os.path.join(HERE, 'LOSO_RidgeRFSVR_fit80_hyperparameters.csv')).assign(Scheme='LOSO')
    hp_all = pd.concat([hp_user, hp_a[['Scheme', 'Model', 'Target', 'Fold', 'Param', 'Value']]], ignore_index=True)
    kf_user = pd.read_csv(os.path.join(HERE, 'KFold_your_runs_v2_combined.csv'))
    lo_user = pd.read_csv(os.path.join(HERE, 'LOSO_your_runs_v2_combined.csv'))
    cases = []
    for t in D:
        for m in ['Ridge', 'RF', 'SVR', 'GB', 'XGB', 'DNN']:
            r = kf_user[(kf_user.Target == t) & (kf_user.Model == m)]
            if len(r) and r.R2.iloc[0] > 0:
                cases.append(('5-fold CV', t, m, float(r.R2.iloc[0])))
        for m in ['XGB', 'GB', 'DNN']:
            r = lo_user[(lo_user.Target == t) & (lo_user.Model == m)]
            if len(r) and r.R2.iloc[0] > 0:
                cases.append(('LOSO', t, m, float(r.R2.iloc[0])))
        for _, r in loso_a[(loso_a.Target == t) & (loso_a.R2 > 0)].iterrows():
            cases.append(('LOSO', t, r.Model, float(r.R2)))
    log(f'[B] {len(cases)} permutation cases:', [c[:3] for c in cases])
    rows, nulls = [], []
    f_rows = os.path.join(HERE, 'Permutation_tests_fit80.csv')
    f_null = os.path.join(HERE, 'Permutation_null_distributions_fit80.csv')
    if (RESUME or RESUME_B) and os.path.exists(f_rows):
        rows = pd.read_csv(f_rows).to_dict('records')
        nulls = pd.read_csv(f_null).to_dict('records')
    done = {(r['Scheme'], r['Target'], r['Model']) for r in rows}
    for ci, (scheme, t, m, r2_reported) in enumerate(cases):
        if (scheme, t, m) in done:
            continue
        rng = np.random.default_rng(SEED + ci)
        X, y, g = D[t]
        splits = kfold_splits(t, len(y)) if scheme == '5-fold CV' else list(LeaveOneGroupOut().split(X, y, g))
        hp = fold_hps(hp_all, scheme, m, t, len(splits))   # the hyperparameters selected in each outer fold
        t0 = time.time()
        real = outer_pred(X, y, splits, m, hp, scheme)
        n = N_PERM[m]
        yps = [rng.permutation(y) for _ in range(n)]
        null = np.array(Parallel(n_jobs=N_JOBS)(delayed(outer_pred)(X, yp, splits, m, hp, scheme) for yp in yps))
        rows.append(dict(Scheme=scheme, Target=t, Model=m, R2_reported=r2_reported, R2_replica=real, n_perm=n,
                         null_mean=null.mean(), null_95pct=np.percentile(null, 95), null_max=null.max(),
                         p_value=(1 + (null >= real).sum()) / (1 + n)))
        nulls += [dict(Scheme=scheme, Target=t, Model=m, perm=i, R2_null=v) for i, v in enumerate(null)]
        log(f'[B] {scheme:9s} {t:4s} {m:6s} reported {r2_reported:6.3f} replica {real:6.3f} | null mean {null.mean():6.2f} '
            f'95% {np.percentile(null, 95):6.2f} | p = {rows[-1]["p_value"]:.3f}  ({time.time() - t0:.0f} s)')
        pd.DataFrame(rows).to_csv(os.path.join(HERE, 'Permutation_tests_fit80.csv'), index=False)
        pd.DataFrame(nulls).to_csv(os.path.join(HERE, 'Permutation_null_distributions_fit80.csv'), index=False)


if __name__ == '__main__':
    D = data()
    log(f'=== run {time.ctime()}  N_PERM {N_PERM}')
    for t, (X, y, g) in D.items():
        log(f'{t}: {len(y)} records, {len(np.unique(g))} studies')
    if RESUME and os.path.exists(os.path.join(HERE, 'LOSO_RidgeRFSVR_fit80_metrics.csv')):
        loso_a = pd.read_csv(os.path.join(HERE, 'LOSO_RidgeRFSVR_fit80_metrics.csv'))
        log('[A] resumed from saved LOSO_RidgeRFSVR_fit80_metrics.csv')
    else:
        loso_a, _ = part_a(D)
    part_b(D, loso_a)
    log('=== done', time.ctime())
