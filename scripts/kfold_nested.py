"""
Leakage-free nested, stratified K-fold evaluation (record level) for the AMPBR models (2nd-round revision).

Complements loso_validation.py: K-fold answers "can the model predict new operating conditions of systems
already represented in the data?", LOSO answers "can it predict a new, unseen study/system?".

For each target (BMP, COD_Efflu, N_Efflu, P_Efflu):
  1. Outer loop: stratified 5-fold CV on target quintiles, repeated 3 times with different shuffles. The
     outer test fold is never used for imputation, scaling, CSI or tuning.
  2. Inner loop (training folds only): mean imputation + standardization (inputs and target) inside a
     pipeline, hyperparameters tuned by random search (20 candidates, 3-fold CV, MSE).
  3. Models: ridge, RF, XGBoost, SVR, and an MLP with the Table 1 architecture of the corresponding DNN.
  4. Baselines: training mean, and a "study-average lookup" that ignores all inputs and predicts each test
     record as the mean target of the training records from the same study (training mean if none).
     A model that does not beat this lookup has learned study identity, not process relations.
  5. CSI comparison (R5.2): EDS only vs EDS + CSI applied to the training folds only (fixed hyperparameters).
  6. Permutation test for the best model: targets shuffled against the inputs, whole nested procedure repeated.
  7. Final model (Table 1): the same tuning procedure run once on all records; one hyperparameter set reported.

R2, MAE, MSE and RMSE are pooled over all out-of-fold predictions of a repeat, then averaged over repeats.
Inputs : Data-AMBR_ori_v3.csv, Table_S1_v3.xlsx (column 'Ref' = study). Output: KFold_nested_results.xlsx.
"""
import os, sys, time, warnings
import numpy as np, pandas as pd
from scipy.stats import loguniform
from sklearn.model_selection import StratifiedKFold, KFold, RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from joblib import Parallel, delayed
import loso_validation as L
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------- settings
BASE = sys.argv[1] if len(sys.argv) > 1 else L.BASE
DATA = os.path.join(BASE, 'Data-AMBR_ori_v3.csv')
TABLE_S1 = os.path.join(BASE, 'Table_S1_v3.xlsx')
OUTER_K, REPEATS, INNER_K, N_ITER = 5, 3, 3, 20
N_PERM = int(sys.argv[2]) if len(sys.argv) > 2 else 50
N_JOBS = -1             # parallel jobs (models x repeats, and permutations)
SEED = 0


def spaces(target):
    return {
        'Ridge': (Ridge(), {'alpha': loguniform(1e-3, 1e3)}),
        'RF': (RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=1),
               {'max_depth': [None, 3, 5, 8], 'min_samples_leaf': [1, 2, 4], 'max_features': [0.3, 0.6, 1.0]}),
        'XGBoost': (XGBRegressor(random_state=SEED, verbosity=0, n_jobs=1),
                    {'n_estimators': [100, 300], 'learning_rate': loguniform(0.01, 0.3), 'max_depth': [2, 3, 4, 6],
                     'subsample': [0.7, 1.0], 'min_child_weight': [1, 3, 5]}),
        'SVR': (SVR(), {'C': loguniform(0.1, 100), 'gamma': loguniform(1e-3, 1), 'epsilon': [0.01, 0.1, 0.3]}),
        'DNN (Table 1 arch.)': (MLPRegressor(hidden_layer_sizes=L.MLP_ARCH[target], max_iter=600, early_stopping=True,
                                             random_state=SEED),
                                {'alpha': loguniform(1e-5, 1e-1), 'learning_rate_init': loguniform(1e-3, 3e-2)}),
    }


def tuned(model, grid, seed):
    """Imputation, scaling and tuning all inside the training data passed to .fit()."""
    pipe = TransformedTargetRegressor(make_pipeline(SimpleImputer(), StandardScaler(), model), transformer=StandardScaler())
    grid = {f'regressor__{model.__class__.__name__.lower()}__{k}': v for k, v in grid.items()}
    return RandomizedSearchCV(pipe, grid, n_iter=N_ITER, cv=KFold(INNER_K, shuffle=True, random_state=seed),
                              scoring='neg_mean_squared_error', random_state=seed, n_jobs=1)


def outer_splits(y, seed):
    bins = pd.qcut(y, OUTER_K, labels=False, duplicates='drop')
    return list(StratifiedKFold(OUTER_K, shuffle=True, random_state=seed).split(np.zeros(len(y)), bins))


def metrics(y, p):
    mse = mean_squared_error(y, p)
    return dict(R2=r2_score(y, p), MAE=mean_absolute_error(y, p), MSE=mse, RMSE=mse ** 0.5)


def nested_pred(X, y, model, grid, splits, seed):
    p = np.empty_like(y)
    for tr, te in splits:
        p[te] = tuned(model, grid, seed).fit(X[tr], y[tr]).predict(X[te])
    return p


def baseline_preds(y, g, splits):
    pm, ps = np.empty_like(y), np.empty_like(y)
    for tr, te in splits:
        pm[te] = y[tr].mean()
        for i in te:
            same = tr[g[tr] == g[i]]
            ps[i] = y[same].mean() if len(same) else y[tr].mean()
    return pm, ps


def csi_pred(X, y, make_model, splits, use_csi):
    p = np.empty_like(y)
    for tr, te in splits:
        imp = SimpleImputer().fit(X[tr]); Xtr, ytr = imp.transform(X[tr]), y[tr]
        if use_csi:
            Xtr, ytr = L.csi_augment(Xtr, ytr)
        p[te] = L.fit_predict(Xtr, ytr, imp.transform(X[te]), make_model)
    return p


def main():
    d = pd.read_csv(DATA)
    d['Study_ID'] = pd.read_excel(TABLE_S1)['Ref'].astype(str).str.strip().str.strip('[]').astype(int).values
    rng = np.random.default_rng(SEED)
    rep, summ, final, csi, perm = [], [], [], [], []
    for tname, tcol in L.TARGETS.items():
        sub = d[d[tcol].notna() & ~d[tcol].round(4).isin(np.round(L.OUTLIERS[tname], 4))]
        X, y, g = sub[L.INPUTS].to_numpy(float), sub[tcol].to_numpy(float), sub['Study_ID'].to_numpy()
        splits = {r: outer_splits(y, SEED + r) for r in range(REPEATS)}
        share = np.mean([(g[tr] == g[i]).any() for tr, te in splits[0] for i in te])
        print(f'\n{tname}: {len(y)} records, {len(np.unique(g))} studies; test records with a same-study record '
              f'in training: {share:.0%}', flush=True)
        for r, sp in splits.items():
            pm, ps = baseline_preds(y, g, sp)
            rep.append(dict(target=tname, repeat=r, model='Mean (null)', **metrics(y, pm)))
            rep.append(dict(target=tname, repeat=r, model='Study-average lookup (no inputs)', **metrics(y, ps)))
        t0 = time.time()
        jobs = [(r, mname) for r in splits for mname in spaces(tname)]
        preds = Parallel(n_jobs=N_JOBS)(delayed(nested_pred)(X, y, *spaces(tname)[m], splits[r], SEED + r) for r, m in jobs)
        for (r, mname), p in zip(jobs, preds):
            rep.append(dict(target=tname, repeat=r, model=mname, **metrics(y, p)))
            print(f'  rep {r} {mname:22s} R2 = {rep[-1]["R2"]:.3f}', flush=True)
        print(f'  ({time.time() - t0:.0f} s)', flush=True)
        df = pd.DataFrame([x for x in rep if x['target'] == tname])
        agg = df.groupby('model', sort=False).agg(R2=('R2', 'mean'), R2_sd=('R2', 'std'), MAE=('MAE', 'mean'),
                                                  MSE=('MSE', 'mean'), RMSE=('RMSE', 'mean')).reset_index()
        agg.insert(0, 'target', tname); agg.insert(1, 'records', len(y)); agg.insert(2, 'same_study_in_training', share)
        summ.append(agg)
        # CSI comparison, fixed hyperparameters, same outer folds (repeat 0)
        for mname, mk in L.models(tname).items():
            if mk is None:
                continue
            a = r2_score(y, csi_pred(X, y, mk, splits[0], False)); b = r2_score(y, csi_pred(X, y, mk, splits[0], True))
            csi.append(dict(target=tname, model=mname, R2_EDS_only=a, R2_EDS_plus_CSI=b, change=b - a))
        # final model: same tuning procedure once on all records
        models_ = agg[~agg.model.isin(['Mean (null)', 'Study-average lookup (no inputs)'])]
        best = models_.sort_values('R2', ascending=False).model.iloc[0]
        for mname, (model, grid) in spaces(tname).items():
            s = tuned(model, grid, SEED).fit(X, y)
            final.append(dict(target=tname, model=mname, best_in_kfold=mname == best,
                              hyperparameters={k.split('__')[-1]: (round(float(v), 5) if isinstance(v, (float, np.floating)) else v)
                                               for k, v in s.best_params_.items()}))
        # permutation test of the best model (whole nested procedure on shuffled targets)
        model, grid = spaces(tname)[best]; real = agg.set_index('model').loc[best, 'R2']; t0 = time.time()
        yps = [rng.permutation(y) for _ in range(N_PERM)]
        null = np.array([r2_score(yp, p) for yp, p in zip(yps, Parallel(n_jobs=N_JOBS)(
            delayed(nested_pred)(X, yp, model, grid, splits[0], SEED) for yp in yps))])
        perm.append(dict(target=tname, model=best, R2=real, n_perm=N_PERM, null_mean=null.mean(),
                         null_95pct=np.percentile(null, 95), p_value=(1 + (null >= real).sum()) / (1 + N_PERM)))
        print(f'  best {best}: R2 {real:.3f}, permutation p = {perm[-1]["p_value"]:.3f} '
              f'(null mean {null.mean():.2f}; {time.time() - t0:.0f} s)', flush=True)
    out = os.path.join(BASE, 'KFold_nested_results.xlsx')
    with pd.ExcelWriter(out) as w:
        pd.concat(summ).round(3).to_excel(w, sheet_name='Summary', index=False)
        pd.DataFrame(rep).round(3).to_excel(w, sheet_name='Repeats', index=False)
        pd.DataFrame(csi).round(3).to_excel(w, sheet_name='CSI comparison', index=False)
        pd.DataFrame(perm).round(3).to_excel(w, sheet_name='Permutation', index=False)
        pd.DataFrame(final).astype(str).to_excel(w, sheet_name='Final hyperparameters', index=False)
    print(pd.concat(summ).round(3).to_string()); print(pd.DataFrame(csi).round(3).to_string())
    print(pd.DataFrame(perm).round(3).to_string()); print('Saved', out)


if __name__ == '__main__':
    main()
