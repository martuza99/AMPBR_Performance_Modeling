"""R1.2 (round 3): LOSO with study-grouped inner validation for hyperparameter tuning.

Outer loop : leave-one-study-out, the same folds and per-fold (imputed) files as the reported LOSO results
             (AMBR Prediction_v5loso/Training_/Testing_data_ori_{T}-fold{k}.csv).
Candidates : in every outer fold, N_CAND configurations are drawn at random from the Table 1 search space of each
             model (fixed seed per fold); the SAME candidates are scored by both inner schemes, so the two arms
             differ only in how the configuration is selected:
   random  : one random 80/20 split of the training records (train_test_split, random_state=42; as in the reported
             runs); scaler fitted on the 80% subset; highest validation R2 selected; model kept as fitted on 80%.
   grouped : GroupKFold over the training STUDIES (5 inner folds; every study in one inner validation fold);
             scaler fitted on each inner training set; highest pooled inner R2 selected; refitted on all
             training records of the outer fold.
DNN        : scikit-learn MLPRegressor, 2 or 4 hidden layers of 8-128 units (step 8), ReLU, Adam, learning rate 0.01,
             max 300 epochs (same implementation in both arms).
Outputs    : LOSO_grouped_inner_predictions.csv, LOSO_grouped_inner_metrics.csv, LOSO_grouped_inner_hyperparameters.csv
Usage      : python loso_grouped_inner.py [test]   (test = one fold per target/model, timing only)
"""
import os, sys, time, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from joblib import Parallel, delayed

warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'
B = r'E:\My Drive\Colab Notebooks\AMBR Prediction_v5loso'
HERE = os.path.dirname(os.path.abspath(__file__))
N_CAND, N_INNER, N_JOBS, SEED = 30, 5, 4, 42
TEST = 'test' in sys.argv
MODELS = ['Ridge', 'SVR', 'DNN', 'XGB', 'GB', 'RF']
LOG = open(os.path.join(HERE, 'loso_grouped_inner.log'), 'a', encoding='utf-8')


def log(s):
    print(s, flush=True); LOG.write(s + '\n'); LOG.flush()


# ------------------------------------------------------------------ search spaces (Table 1)
def sample(m, rng):
    u, ri, lg = rng.uniform, rng.integers, lambda a, b: 10 ** rng.uniform(a, b)
    if m == 'Ridge': return dict(alpha=lg(-4, 4))
    if m == 'SVR': return dict(C=lg(-2, 3), epsilon=lg(-3, 0), gamma=lg(-4, 1))
    if m == 'RF': return dict(n_estimators=int(ri(100, 1001)), max_depth=int(ri(2, 31)), min_samples_split=int(ri(2, 21)),
                              min_samples_leaf=int(ri(1, 11)), max_features=float(u(0.3, 1.0)))
    if m == 'GB': return dict(n_estimators=int(ri(100, 1501)), learning_rate=float(u(1e-3, 0.5)), max_depth=int(ri(1, 11)),
                              min_samples_split=int(ri(2, 21)), min_samples_leaf=int(ri(1, 11)),
                              subsample=float(u(0.5, 1.0)), max_features=float(u(0.3, 1.0)))
    if m == 'XGB': return dict(n_estimators=int(ri(100, 1501)), learning_rate=float(u(1e-5, 1.0)), max_depth=int(ri(1, 11)),
                               min_child_weight=float(u(1, 10)), subsample=float(u(0.5, 1.0)),
                               colsample_bytree=float(u(0.5, 1.0)))
    if m == 'DNN':
        depth = 2 if rng.random() < 0.5 else 4
        return dict(units='-'.join(str(int(8 * ri(1, 17))) for _ in range(depth)))
    raise ValueError(m)


def make(m, h):
    if m == 'Ridge': return Ridge(alpha=h['alpha'])
    if m == 'SVR': return SVR(kernel='rbf', C=h['C'], epsilon=h['epsilon'], gamma=h['gamma'])
    if m == 'RF': return RandomForestRegressor(**h, random_state=SEED, n_jobs=1)
    if m == 'GB': return GradientBoostingRegressor(**h, random_state=SEED)
    if m == 'XGB': return XGBRegressor(**h, random_state=SEED, n_jobs=1, verbosity=0)
    if m == 'DNN': return MLPRegressor(hidden_layer_sizes=tuple(int(x) for x in h['units'].split('-')), activation='relu',
                                       solver='adam', learning_rate_init=0.01, max_iter=300, random_state=SEED)


def fit_predict(m, h, Xa, ya, Xb_list):
    """Scale inputs and target with the fitting data, fit, return predictions (physical units) for each Xb."""
    sx, sy = StandardScaler().fit(Xa), StandardScaler().fit(ya[:, None])
    est = make(m, h).fit(sx.transform(Xa), sy.transform(ya[:, None]).ravel())
    return [sy.inverse_transform(est.predict(sx.transform(Xb))[:, None]).ravel() for Xb in Xb_list]


def one_fold(T, k, m, study_out):
    tr = pd.read_csv(os.path.join(B, f'Training_data_ori_{T}-fold{k}.csv'), encoding='latin-1')
    te = pd.read_csv(os.path.join(B, f'Testing_data_ori_{T}-fold{k}.csv'), encoding='latin-1')
    s1 = pd.read_excel(os.path.join(B, f'Table_S1_v3_pp_num_{T}.xlsx'))
    g = s1.loc[s1.Ref.astype(str).str.strip() != study_out, 'Ref'].astype(str).str.strip().to_numpy()
    X, y = tr.iloc[:, :-1].to_numpy(float), tr.iloc[:, -1].to_numpy(float)
    Xt, yt = te.iloc[:, :-1].to_numpy(float), te.iloc[:, -1].to_numpy(float)
    assert len(g) == len(y)
    rng = np.random.default_rng(1000 * ['BMP', 'COD', 'NR', 'PR'].index(T) + k)
    cands = [sample(m, rng) for _ in range(N_CAND)]

    fi, vi = train_test_split(np.arange(len(y)), test_size=0.2, random_state=42, shuffle=True)
    inner = list(GroupKFold(n_splits=min(N_INNER, len(np.unique(g)))).split(X, y, g))
    sc_r, sc_g = [], []
    for h in cands:
        sc_r.append(r2_score(y[vi], fit_predict(m, h, X[fi], y[fi], [X[vi]])[0]))
        p = np.empty_like(y)
        for a, b in inner:
            p[b] = fit_predict(m, h, X[a], y[a], [X[b]])[0]
        sc_g.append(r2_score(y, p))
    hr, hg = cands[int(np.nanargmax(sc_r))], cands[int(np.nanargmax(sc_g))]
    pr = fit_predict(m, hr, X[fi], y[fi], [Xt])[0]      # random arm: kept as fitted on the 80% subset
    pg = fit_predict(m, hg, X, y, [Xt])[0]              # grouped arm: refitted on all training records
    return dict(T=T, k=k, m=m, study=study_out, y=yt, pr=pr, pg=pg, hr=hr, hg=hg,
                val_r=max(sc_r), val_g=max(sc_g))


def metr(y, p):
    return dict(R2=r2_score(y, p), MAE=mean_absolute_error(y, p), RMSE=mean_squared_error(y, p) ** 0.5)


if __name__ == '__main__':
    fmap = pd.read_csv(os.path.join(B, 'LOSO_fold_to_study.csv'))
    f_pred, f_met, f_hp = (os.path.join(HERE, f'LOSO_grouped_inner_{x}.csv') for x in ('predictions', 'metrics', 'hyperparameters'))
    done = set()
    P, M, H = [], [], []
    if not TEST and os.path.exists(f_met):
        M = pd.read_csv(f_met).to_dict('records'); P = pd.read_csv(f_pred).to_dict('records'); H = pd.read_csv(f_hp).to_dict('records')
        done = {(r['Target'], r['Model']) for r in M}
    log(f'=== run {time.ctime()}  N_CAND={N_CAND} N_INNER={N_INNER} test={TEST}')
    for m in MODELS:
        for T in ['BMP', 'COD', 'NR', 'PR']:
            if (T, m) in done: continue
            fm = fmap[fmap.target == T].sort_values('fold')
            jobs = [(int(r.fold), str(r.Ref).strip()) for r in fm.itertuples()]
            if TEST: jobs = jobs[:1]
            t0 = time.time()
            res = Parallel(n_jobs=1 if TEST else N_JOBS)(delayed(one_fold)(T, k, m, s) for k, s in jobs)
            if TEST:
                log(f'[test] {T} {m}: one fold {time.time() - t0:.1f} s'); continue
            y = np.concatenate([r['y'] for r in res]); pr = np.concatenate([r['pr'] for r in res])
            pg = np.concatenate([r['pg'] for r in res])
            for r in res:
                P += [dict(Target=T, Model=m, Fold=r['k'], Study_ID=r['study'], Actual=a, Pred_random=b, Pred_grouped=c)
                      for a, b, c in zip(r['y'], r['pr'], r['pg'])]
                for arm, h in (('random', r['hr']), ('grouped', r['hg'])):
                    H += [dict(Target=T, Model=m, Fold=r['k'], Study_ID=r['study'], Arm=arm, Param=p, Value=v) for p, v in h.items()]
            a, b = metr(y, pr), metr(y, pg)
            M.append(dict(Target=T, Model=m, N=len(y), Studies=len(res), **{f'{k}_random': v for k, v in a.items()},
                          **{f'{k}_grouped': v for k, v in b.items()},
                          same_config_folds=sum(r['hr'] == r['hg'] for r in res)))
            log(f'{T:4s} {m:6s} LOSO R2 random-inner {a["R2"]:7.3f} | grouped-inner {b["R2"]:7.3f} '
                f'(MAE {a["MAE"]:.3f} -> {b["MAE"]:.3f})  {time.time() - t0:.0f} s')
            pd.DataFrame(P).to_csv(f_pred, index=False); pd.DataFrame(M).to_csv(f_met, index=False)
            pd.DataFrame(H).to_csv(f_hp, index=False)
    log('=== done ' + time.ctime())
