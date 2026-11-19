"""
Leakage-free, study-grouped validation for the AMPBR models (2nd-round revision: R1.1, R5.1, R5.2, R5.3).

For each target (BMP, COD_Efflu, N_Efflu, P_Efflu):
  1. Leave-one-study-out (LOSO) cross-validation: all records of one source study are held out;
     mean imputation, scaling and (optionally) CSI are fitted on the training studies only.
  2. Models: mean predictor (null baseline), ridge regression, RF, XGBoost, SVR, and an MLP with the
     Table 1 architecture of the corresponding DNN. Hyperparameters are fixed in advance (no tuning
     inside the folds), so no information from the held-out study enters model selection.
  3. Two training variants: EDS only, and EDS + CSI (manuscript procedure, applied to the training
     studies only: sort by target, rank index, natural cubic splines, 500 query points, clip).
  4. Permutation (null) tests for models with positive LOSO R2: the target is shuffled against the
     inputs and the whole LOSO procedure is repeated; p = (1 + #null R2 >= real R2) / (1 + n_perm).
  5. Null check of the ORIGINAL procedure (CSI on the complete EDS, then 5-fold CV on the IDS, and the
     EDS consistency check), with real and with permuted targets.

R2, MAE and RMSE are POOLED: all held-out predictions are collected and the metric is computed once
over all records (per-study R2 is undefined for single-record studies).

Inputs : Data-AMBR_ori_v3.csv (raw EDS with missing values) and Table_S1_v3.xlsx (column 'Ref' = study).
Outputs: LOSO_results.csv, LOSO_permutation.csv, Original_procedure_null_check.csv (in BASE).
Runtime: about 10-30 min on Colab (most of it in the permutation tests; reduce N_PERM to go faster).
"""
import os, warnings
import numpy as np, pandas as pd
from scipy.interpolate import CubicSpline
from sklearn.model_selection import LeaveOneGroupOut, KFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
warnings.filterwarnings('ignore')
try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

# ---------------------------------------------------------------- settings
BASE = r'E:\My Drive\Colab Notebooks\AMBR Prediction_v4'
DATA = os.path.join(BASE, 'Data-AMBR_ori_v3.csv')
TABLE_S1 = os.path.join(BASE, 'Table_S1_v3.xlsx')      # 'Ref' column gives the source study of each record
N_CSI = 500             # interpolated records per target (as in the manuscript)
N_PERM = 200            # permutations per model (models with positive LOSO R2 only)
N_PERM_SLOW = 50        # permutations for RF / XGBoost / MLP and for CSI variants
SEED = 0
INPUTS = ['COD (mg/L)', 'N (mg/L)', 'P (mg/L)', 'pH', 'Avg. Temperature 0C', 'Flux (L/m2/h)',
          'HRT (d)', 'SRT (d)', 'Light intensity (umol/m2s)', 'Light cycle (h)']
TARGETS = {'BMP': 'Productivity (mg/L·d)', 'COD_Efflu': 'Effluent COD  (mg/L)',
           'N_Efflu': 'Effluent N  (mg/L)', 'P_Efflu': 'Effluent P  (mg/L)'}
# All records are kept (no outlier removal). To drop records, list their target values here,
# e.g. {'BMP': [230.0, 158.1818182], 'COD_Efflu': [136.5], ...}.
OUTLIERS = {'BMP': [], 'COD_Efflu': [], 'N_Efflu': [], 'P_Efflu': []}
AMBIENT_T = 23.0        # unreported reactor temperature is taken as ambient 23 °C (assumption, applied before any split)
MLP_ARCH = {'BMP': (8, 248, 136, 168), 'COD_Efflu': (40, 128, 112, 128),
            'N_Efflu': (120, 48, 16, 40), 'P_Efflu': (112, 32, 24, 128)}


def models(target):
    m = {'Mean (null)': None,
         'Ridge': lambda: Ridge(alpha=1.0),
         'RF': lambda: RandomForestRegressor(n_estimators=500, random_state=SEED, n_jobs=-1),
         'GB': lambda: GradientBoostingRegressor(random_state=SEED),
         'SVR': lambda: SVR(),
         'DNN (Table 1 arch.)': lambda: MLPRegressor(hidden_layer_sizes=MLP_ARCH[target], learning_rate_init=0.01,
                                                     max_iter=600, early_stopping=True, random_state=SEED)}
    if XGBRegressor is not None:
        m['XGBoost'] = lambda: XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=SEED, verbosity=0)
    return m


def csi_augment(X, y, n=N_CSI):
    """Manuscript CSI: sort by y, rank index t, natural cubic splines of each column vs t, n query points, clip."""
    order = np.argsort(y, kind='stable')
    Xs, ys = X[order], y[order]
    t = np.arange(len(ys), dtype=float)
    tq = np.linspace(0, len(ys) - 1, n)
    Xq = np.column_stack([CubicSpline(t, Xs[:, k], bc_type='natural')(tq) for k in range(X.shape[1])])
    yq = CubicSpline(t, ys, bc_type='natural')(tq)
    return np.clip(Xq, X.min(0), X.max(0)), np.clip(yq, y.min(), y.max())


def fit_predict(Xtr, ytr, Xte, make_model):
    if make_model is None:
        return np.full(len(Xte), ytr.mean())
    sx = StandardScaler().fit(Xtr); mu, sd = ytr.mean(), ytr.std() or 1.0
    mdl = make_model().fit(sx.transform(Xtr), (ytr - mu) / sd)
    return mdl.predict(sx.transform(Xte)) * sd + mu


def loso_predict(X, y, groups, make_model, use_csi):
    pred = np.empty_like(y)
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        imp = SimpleImputer(strategy='mean').fit(X[tr])
        Xtr, Xte, ytr = imp.transform(X[tr]), imp.transform(X[te]), y[tr]
        if use_csi:
            Xtr, ytr = csi_augment(Xtr, ytr)
        pred[te] = fit_predict(Xtr, ytr, Xte, make_model)
    return pred


def original_procedure(X, y):
    """Manuscript procedure: impute + CSI on the complete EDS, 5-fold CV on the IDS, then EDS consistency check (RF)."""
    X = SimpleImputer().fit_transform(X)
    Xi, yi = csi_augment(X, y)
    pv = np.empty_like(yi)
    for tr, te in KFold(5, shuffle=True, random_state=SEED).split(Xi):
        pv[te] = fit_predict(Xi[tr], yi[tr], Xi[te], models('BMP')['RF'])
    return r2_score(yi, pv), r2_score(y, fit_predict(Xi, yi, X, models('BMP')['RF']))


def load():
    d = pd.read_csv(DATA)
    s1 = pd.read_excel(TABLE_S1)
    assert len(s1) == len(d), 'Table S1 and the data file must have the same records in the same order'
    d['Study_ID'] = s1['Ref'].astype(str).str.strip().str.strip('[]').astype(int).values
    d['Avg. Temperature 0C'] = d['Avg. Temperature 0C'].fillna(AMBIENT_T)
    return d


def main():
    d = load()
    rng = np.random.default_rng(SEED)
    res, perm, null = [], [], []
    for tname, tcol in TARGETS.items():
        sub = d[d[tcol].notna() & ~d[tcol].round(4).isin(np.round(OUTLIERS[tname], 4))]
        X, y, g = sub[INPUTS].to_numpy(float), sub[tcol].to_numpy(float), sub['Study_ID'].to_numpy()
        print(f'\n{tname}: {len(y)} records from {len(np.unique(g))} studies')
        a, b = original_procedure(X, y)
        yp = rng.permutation(y); c, e = original_procedure(X, yp)
        null.append(dict(target=tname, IDS_CV_R2_real=a, EDS_R2_real=b, IDS_CV_R2_permuted=c, EDS_R2_permuted=e))
        print(f'  original procedure (RF): real IDS-CV {a:.2f} / EDS {b:.2f} | permuted IDS-CV {c:.2f} / EDS {e:.2f}')
        for mname, mk in models(tname).items():
            for use_csi in ([False] if mk is None else [False, True]):
                p = loso_predict(X, y, g, mk, use_csi)
                r2 = r2_score(y, p)
                row = dict(target=tname, records=len(y), studies=len(np.unique(g)), model=mname,
                           training='EDS + CSI' if use_csi else 'EDS only', R2=r2,
                           MAE=mean_absolute_error(y, p), RMSE=mean_squared_error(y, p) ** 0.5)
                res.append(row)
                print(f'  {mname:22s} {row["training"]:10s} R2 = {r2:6.3f}  MAE = {row["MAE"]:.2f}', flush=True)
                if mk is None or r2 <= 0:
                    continue
                n = N_PERM if (mname in ('Ridge', 'SVR') and not use_csi) else N_PERM_SLOW
                nr = np.array([r2_score(yp, loso_predict(X, yp, g, mk, use_csi)) for yp in (rng.permutation(y) for _ in range(n))])
                perm.append(dict(target=tname, model=mname, training=row['training'], R2=r2, n_perm=n,
                                 null_mean=nr.mean(), null_95pct=np.percentile(nr, 95), p_value=(1 + (nr >= r2).sum()) / (1 + n)))
                print(f'      permutation p = {perm[-1]["p_value"]:.3f} (null mean {nr.mean():.2f})', flush=True)
    pd.DataFrame(res).round(3).to_csv(os.path.join(BASE, 'LOSO_results.csv'), index=False)
    pd.DataFrame(perm).round(3).to_csv(os.path.join(BASE, 'LOSO_permutation.csv'), index=False)
    pd.DataFrame(null).round(3).to_csv(os.path.join(BASE, 'Original_procedure_null_check.csv'), index=False)
    print('\nSaved LOSO_results.csv, LOSO_permutation.csv, Original_procedure_null_check.csv')


if __name__ == '__main__':
    main()

