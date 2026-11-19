"""Sobol global sensitivity analysis (S1st, S2nd, ST) for effluent P with the XGB models.
Primary: the five 5-fold CV XGB models saved by the user (v5kcv2), each with its own scaler rebuilt exactly as in the
user's code (train_test_split 0.2, random_state=42; StandardScaler on the 80% fitting subset); the check reproduces the
saved test predictions. Indices are reported as the mean over the five models, with min-max across models.
Check: one final XGB tuned once on all 68 records with the same two-stage bayes_opt procedure (20 + 200 per stage).
Inputs sampled independently and uniformly over the observed range of the 68 records (Saltelli / Sobol sequence)."""
import os, sys, glob, warnings, time
import numpy as np, pandas as pd, joblib, xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol as sobol_analyze
from bayes_opt import BayesianOptimization

warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
K = r'E:\My Drive\Colab Notebooks\AMBR Prediction_v5kcv2'
MD = glob.glob(os.path.join(K, 'XGBoost PR Best Models_*'))[0]
NAMES = ['CODInflu', 'NInflu', 'PInflu', 'pH', 'Temp', 'Flux', 'HRT', 'SRT', 'Light', 'Photoperiod']
NS = [512, 1024, 2048, 4096, 8192]
NMAIN = 4096
SKIP_FINAL = 'nofinal' in sys.argv


def rd(f):
    return pd.read_csv(f, encoding='latin-1')


# ---------------------------------------------------------------- the five fold models
models = {}
for k in range(1, 6):
    tr = rd(os.path.join(K, f'Training_data_ori_PR-fold{k}.csv'))
    te = rd(os.path.join(K, f'Testing_data_ori_PR-fold{k}.csv'))
    fit, _ = train_test_split(tr, test_size=0.20, random_state=42, shuffle=True)
    sc = StandardScaler().fit(fit)
    m = joblib.load(glob.glob(os.path.join(MD, f'XGB_model_PR_kote_{k}.joblib'))[0])
    ref = pd.read_csv(os.path.join(MD, f'Predictions_PR_kote_{k}.csv'))['Prediction(real)'].to_numpy()
    p = m.predict(sc.transform(te)[:, :10]) * sc.scale_[-1] + sc.mean_[-1]
    print(f'fold {k}: reproduction max |diff| = {np.abs(p - ref).max():.2e}')
    models[f'fold{k}'] = (m, sc)

alld = pd.concat([rd(os.path.join(K, 'Training_data_ori_PR-fold1.csv')),
                  rd(os.path.join(K, 'Testing_data_ori_PR-fold1.csv'))], ignore_index=True)
assert len(alld) == 68, len(alld)
X = alld.iloc[:, :10].to_numpy(float); Y = alld.iloc[:, 10].to_numpy(float)

# ---------------------------------------------------------------- final model on all records (check)
if not SKIP_FINAL:
    t0 = time.time()
    fit, val = train_test_split(alld, test_size=0.20, random_state=42, shuffle=True)
    sc = StandardScaler().fit(fit)
    d1, d3 = sc.transform(fit), sc.transform(val)
    Xf, yf, Xv, yv = d1[:, :10], d1[:, 10], d3[:, :10], d3[:, 10]
    mk = lambda p: xgb.XGBRegressor(objective='reg:squarederror', n_jobs=2, **p)

    def o1(n_estimators, learning_rate):
        return r2_score(yv, mk(dict(n_estimators=int(n_estimators), learning_rate=learning_rate)).fit(Xf, yf).predict(Xv))
    b1 = BayesianOptimization(f=o1, pbounds={'n_estimators': (100, 1500), 'learning_rate': (1e-5, 1)}, random_state=1, verbose=0)
    b1.maximize(init_points=20, n_iter=200)
    ne, lr = int(b1.max['params']['n_estimators']), float(b1.max['params']['learning_rate'])

    def o2(max_depth, min_child_weight, subsample, colsample_bytree):
        return r2_score(yv, mk(dict(n_estimators=ne, learning_rate=lr, max_depth=int(max_depth),
                                    min_child_weight=int(min_child_weight), subsample=subsample,
                                    colsample_bytree=colsample_bytree)).fit(Xf, yf).predict(Xv))
    b2 = BayesianOptimization(f=o2, pbounds={'max_depth': (1, 10), 'min_child_weight': (1, 10), 'subsample': (0.5, 1.0),
                                             'colsample_bytree': (0.5, 1.0)}, random_state=1, verbose=0)
    b2.maximize(init_points=20, n_iter=200)
    bp = b2.max['params']
    fp = dict(n_estimators=ne, learning_rate=lr, max_depth=int(bp['max_depth']), min_child_weight=int(bp['min_child_weight']),
              subsample=float(bp['subsample']), colsample_bytree=float(bp['colsample_bytree']))
    m = mk(fp).fit(Xf, yf)
    joblib.dump((m, sc), os.path.join(HERE, 'XGB_PR_final_all_records.joblib'))
    pd.DataFrame([dict(**fp, val_R2=b2.max['target'])]).to_csv(os.path.join(HERE, 'XGB_PR_final_hyperparameters.csv'), index=False)
    print(f'final model: {fp}, validation R2 = {b2.max["target"]:.3f} ({time.time() - t0:.0f} s)')
    models['final'] = (m, sc)

# ---------------------------------------------------------------- Sobol
if SKIP_FINAL and os.path.exists(os.path.join(HERE, 'XGB_PR_final_all_records.joblib')):
    models['final'] = joblib.load(os.path.join(HERE, 'XGB_PR_final_all_records.joblib'))
    print('final model loaded from XGB_PR_final_all_records.joblib')
# np.array names/groups: SALib 1.5.1 passes a list to pd.unique, which newer pandas rejects
problem = dict(num_vars=10, names=np.array(NAMES), bounds=[[float(X[:, j].min()), float(X[:, j].max())] for j in range(10)])


def f(model, sc, Xs):
    Z = (Xs - sc.mean_[:10]) / sc.scale_[:10]
    return model.predict(Z) * sc.scale_[-1] + sc.mean_[-1]


conv, s12, s2r, diag = [], [], [], []
for N in NS:
    Xs = sobol_sample.sample(problem, N, calc_second_order=True, seed=1)
    for name, (m, sc) in models.items():
        Ys = f(m, sc, Xs)
        r = sobol_analyze.analyze(problem, Ys, calc_second_order=True, conf_level=0.95, seed=1)
        for j, v in enumerate(NAMES):
            conv.append(dict(Model=name, N=N, Input=v, S1=r['S1'][j], S1_conf=r['S1_conf'][j], ST=r['ST'][j], ST_conf=r['ST_conf'][j]))
        if N == NMAIN:
            for j, v in enumerate(NAMES):
                s12.append(dict(Model=name, Input=v, S1=r['S1'][j], S1_conf=r['S1_conf'][j], ST=r['ST'][j], ST_conf=r['ST_conf'][j]))
                for i in range(j + 1, 10):
                    s2r.append(dict(Model=name, Input_i=v, Input_j=NAMES[i], S2=r['S2'][j, i], S2_conf=r['S2_conf'][j, i]))
            diag.append(dict(Model=name, sum_S1=r['S1'].sum(), frac_pred_PEfflu_gt_PInflu=float(np.mean(Ys > Xs[:, 2])),
                             frac_pred_negative=float(np.mean(Ys < 0)), Y_var=float(Ys.var())))
    print(f'N = {N} done ({len(Xs)} evaluations per model)')

conv, s12, s2r, diag = map(pd.DataFrame, (conv, s12, s2r, diag))
conv.to_csv(os.path.join(HERE, 'Sobol_PR_convergence.csv'), index=False)
s12.to_csv(os.path.join(HERE, 'Sobol_PR_S1_ST_by_model.csv'), index=False)
s2r.to_csv(os.path.join(HERE, 'Sobol_PR_S2_by_model.csv'), index=False)
diag.to_csv(os.path.join(HERE, 'Sobol_PR_diagnostics.csv'), index=False)

folds = s12[s12.Model.str.startswith('fold')]
summ = folds.groupby('Input', sort=False).agg(S1_mean=('S1', 'mean'), S1_min=('S1', 'min'), S1_max=('S1', 'max'),
                                              ST_mean=('ST', 'mean'), ST_min=('ST', 'min'), ST_max=('ST', 'max')).reset_index()
if 'final' in models:
    fm = s12[s12.Model == 'final'].set_index('Input')
    summ['S1_final'] = summ.Input.map(fm.S1); summ['ST_final'] = summ.Input.map(fm.ST)
summ = summ.sort_values('ST_mean', ascending=False)
summ.to_csv(os.path.join(HERE, 'Sobol_PR_S1_ST_summary.csv'), index=False)
f2 = s2r[s2r.Model.str.startswith('fold')]
s2s = f2.groupby(['Input_i', 'Input_j']).agg(S2_mean=('S2', 'mean'), S2_min=('S2', 'min'), S2_max=('S2', 'max'),
                                              conf_mean=('S2_conf', 'mean')).reset_index().sort_values('S2_mean', ascending=False)
s2s.to_csv(os.path.join(HERE, 'Sobol_PR_S2_summary.csv'), index=False)
print(summ.round(3).to_string(index=False)); print(s2s.head(8).round(3).to_string(index=False)); print(diag.round(3).to_string(index=False))
