"""LOSO Ridge / RF / SVR with exactly the user's tuning procedure (from AMBR Prediction_v5/Code/Untitled-1.ipynb):
  data      : the user's per-fold LOSO files (Training_/Testing_data_ori_{T}-fold{k}.csv, imputed per fold);
  split     : train_test_split(training, test_size=0.20, random_state=42) -> 80% fit / 20% validation;
  scaling   : StandardScaler fitted on the 80% fit subset (inputs and target), applied to all three sets;
  tuning    : bayes_opt BayesianOptimization(random_state=1), objective = validation R2,
              Ridge/SVR init_points=20, n_iter=200; RF init_points=10, n_iter=100; user's pbounds;
  final     : best parameters refitted on the 80% fit subset only (no refit on 100%), random_state=42.
Writes the LOSO_RidgeRFSVR_fit80_* files used by round2_fit80.py (Part B permutation tests)."""
import os, time, warnings
import numpy as np, pandas as pd
from bayes_opt import BayesianOptimization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from joblib import Parallel, delayed

warnings.filterwarnings('ignore')
B = r'E:\My Drive\Colab Notebooks\AMBR Prediction_v5loso'
HERE = os.path.dirname(os.path.abspath(__file__))
LOG = open(os.path.join(HERE, 'round2_bo_loso.log'), 'a', encoding='utf-8')
N_JOBS = 4


def log(s):
    print(s, flush=True); LOG.write(s + '\n'); LOG.flush()


SPEC = {
    'Ridge': dict(initp=20, niter=200, pb={'log_alpha': (-4, 4)},
                  make=lambda p, j: Ridge(alpha=10 ** p['log_alpha'], random_state=42),
                  nat=lambda p: {'alpha': 10 ** p['log_alpha']}),
    'SVR': dict(initp=20, niter=200, pb={'log_c': (-2, 3), 'log_epsilon': (-3, 0), 'log_gamma': (-4, 1)},
                make=lambda p, j: SVR(kernel='rbf', C=10 ** p['log_c'], epsilon=10 ** p['log_epsilon'],
                                      gamma=10 ** p['log_gamma']),
                nat=lambda p: {'C': 10 ** p['log_c'], 'epsilon': 10 ** p['log_epsilon'], 'gamma': 10 ** p['log_gamma']}),
    'RF': dict(initp=10, niter=100,
               pb={'n_estimators': (100, 1000), 'max_depth': (2, 30), 'min_samples_split': (2, 20),
                   'min_samples_leaf': (1, 10), 'max_features': (0.3, 1.0)},
               make=lambda p, j: RandomForestRegressor(
                   n_estimators=int(p['n_estimators']), max_depth=int(p['max_depth']),
                   min_samples_split=int(p['min_samples_split']), min_samples_leaf=int(p['min_samples_leaf']),
                   max_features=float(p['max_features']), random_state=42, n_jobs=j),
               nat=lambda p: {'n_estimators': int(p['n_estimators']), 'max_depth': int(p['max_depth']),
                              'min_samples_split': int(p['min_samples_split']),
                              'min_samples_leaf': int(p['min_samples_leaf']), 'max_features': float(p['max_features'])}),
}


def one_fold(T, k, mname):
    sp = SPEC[mname]
    tr = pd.read_csv(os.path.join(B, f'Training_data_ori_{T}-fold{k}.csv'), encoding='latin-1')
    te = pd.read_csv(os.path.join(B, f'Testing_data_ori_{T}-fold{k}.csv'), encoding='latin-1')
    fit, val = train_test_split(tr, test_size=0.20, random_state=42, shuffle=True)
    sc = StandardScaler().fit(fit)
    d1, d2, d3 = sc.transform(fit), sc.transform(te), sc.transform(val)
    Xf, yf, Xt, Xv, yv = d1[:, :-1], d1[:, -1], d2[:, :-1], d3[:, :-1], d3[:, -1]

    def obj(**p):
        return r2_score(yv, sp['make'](p, 1).fit(Xf, yf).predict(Xv))

    opt = BayesianOptimization(f=obj, pbounds=sp['pb'], random_state=1, verbose=0)
    opt.maximize(init_points=sp['initp'], n_iter=sp['niter'])
    best = opt.max['params']
    m = sp['make'](best, 1).fit(Xf, yf)
    ym, ys = sc.mean_[-1], sc.scale_[-1]
    return dict(k=k, actual=te.iloc[:, -1].to_numpy(float), pred=m.predict(Xt) * ys + ym,
                fit_actual=fit.iloc[:, -1].to_numpy(float), fit_pred=m.predict(Xf) * ys + ym,
                hp=sp['nat'](best), val_r2=opt.max['target'])


def metr(y, p):
    return dict(R2=r2_score(y, p), MAE=mean_absolute_error(y, p), RMSE=mean_squared_error(y, p) ** 0.5)


if __name__ == '__main__':
    fmap = pd.read_csv(os.path.join(B, 'LOSO_fold_to_study.csv'))
    preds, mets, hps, trp = [], [], [], []
    log(f'=== run {time.ctime()}')
    for T in ['BMP', 'COD', 'NR', 'PR']:
        fm = fmap[fmap.target == T].sort_values('fold')
        study = {int(r.fold): int(str(r.Ref).strip('[]')) for r in fm.itertuples()}
        for mname in ['Ridge', 'SVR', 'RF']:
            t0 = time.time()
            res = Parallel(n_jobs=N_JOBS)(delayed(one_fold)(T, k, mname) for k in sorted(study))
            A, P, FA, FP = [], [], [], []
            for r in res:
                s = study[r['k']]
                preds.append(pd.DataFrame(dict(Target=T, Model=mname, Fold=r['k'], Study_ID=s,
                                               Actual=r['actual'], Predicted=r['pred'])))
                trp.append(pd.DataFrame(dict(Target=T, Model=mname, Fold=r['k'], Study_out=s,
                                             Actual=r['fit_actual'], Predicted=r['fit_pred'])))
                hps += [dict(Target=T, Model=mname, Fold=r['k'], Study_ID=s, Param=p, Value=v) for p, v in r['hp'].items()]
                A.append(r['actual']); P.append(r['pred']); FA.append(r['fit_actual']); FP.append(r['fit_pred'])
            A, P, FA, FP = map(np.concatenate, (A, P, FA, FP))
            mets.append(dict(Target=T, Model=mname, N=len(A), Studies=len(study), **metr(A, P),
                             **{f'Train_{a}': b for a, b in metr(FA, FP).items()}, N_train_rows=len(FA)))
            log(f'[A-BO] LOSO {T:4s} {mname:6s} test R2 = {mets[-1]["R2"]:7.3f}  MAE = {mets[-1]["MAE"]:.3f} | '
                f'train R2 = {mets[-1]["Train_R2"]:.3f}  ({time.time() - t0:.0f} s)')
            pd.concat(preds).to_csv(os.path.join(HERE, 'LOSO_RidgeRFSVR_fit80_predictions.csv'), index=False)
            pd.DataFrame(mets).to_csv(os.path.join(HERE, 'LOSO_RidgeRFSVR_fit80_metrics.csv'), index=False)
            pd.DataFrame(hps).to_csv(os.path.join(HERE, 'LOSO_RidgeRFSVR_fit80_hyperparameters.csv'), index=False)
            pd.concat(trp).to_csv(os.path.join(HERE, 'LOSO_RidgeRFSVR_fit80_training_predictions.csv'), index=False)
    log('=== done ' + time.ctime())
