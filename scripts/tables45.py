"""Tables 4 (nested 5-fold CV) and 5 (LOSO): training and testing metrics of all models, baselines, permutation p.
Sources: KFold_train_vs_test_metrics.csv, KFold_baselines_your_folds.csv, LOSO_train_vs_test_metrics.csv (XGB/GB/DNN),
LOSO_RidgeRFSVR_fit80_metrics.csv (Ridge/RF/SVR, user's procedure), Permutation_tests_final.csv."""
import os
import numpy as np, pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Font, Border, Side, Alignment

HERE = os.path.dirname(os.path.abspath(__file__))
rd = lambda f: pd.read_csv(os.path.join(HERE, f))
T = ['BMP', 'COD', 'NR', 'PR']
TN = {'BMP': 'BMP (mg/L·d)', 'COD': 'COD_Efflu (mg/L)', 'NR': 'N_Efflu (mg/L)', 'PR': 'P_Efflu (mg/L)'}
M = ['Ridge', 'RF', 'SVR', 'GB', 'XGB', 'DNN']
perm = rd('Permutation_tests_final.csv')
pv = {(r.Scheme, r.Target, r.Model): r.p_value for r in perm.itertuples()}
npm = {(r.Scheme, r.Target, r.Model): r.n_perm for r in perm.itertuples()}

# ---------------- Table 4: 5-fold CV
k = rd('KFold_training_from_saved_models.csv'); kb = rd('KFold_baselines_your_folds.csv')   # recomputed from saved models
r4 = []
for t in T:
    for m in M:
        x = k[(k.Target == t) & (k.Model == m)].iloc[0]
        r4.append(dict(Target=t, Model=m, Train_R2=x.fit_R2, Test_R2=x.test_R2, MAE=x.test_MAE, RMSE=x.test_RMSE,
                       p=pv.get(('5-fold CV', t, m)), n_perm=npm.get(('5-fold CV', t, m))))
    for b, lab in [('Mean predictor', 'Mean predictor'), ('Study-average lookup (no inputs)', 'Study-average baseline')]:
        x = kb[(kb.Target == t) & (kb.Baseline == b)].iloc[0]
        r4.append(dict(Target=t, Model=lab, Test_R2=x.R2, MAE=x.MAE, RMSE=x.RMSE))
t4 = pd.DataFrame(r4)

# ---------------- Table 5: LOSO
lo = rd('LOSO_train_vs_test_metrics.csv'); lr = rd('LOSO_RidgeRFSVR_fit80_metrics.csv')
lg = rd('LOSO_XGBGB_training_from_saved_models.csv')
pr = rd('LOSO_RidgeRFSVR_fit80_predictions.csv')
r5 = []
for t in T:
    for m in M:
        if m in ('Ridge', 'RF', 'SVR'):
            x = lr[(lr.Target == t) & (lr.Model == m)].iloc[0]
            d = dict(Train_R2=x.Train_R2, Test_R2=x.R2, MAE=x.MAE, RMSE=x.RMSE)
        else:
            x = lo[(lo.Target == t) & (lo.Model == m)].iloc[0]
            d = dict(Train_R2=x.Train_R2_overall, Test_R2=x.Test_R2, MAE=x.Test_MAE, RMSE=x.Test_RMSE)
            if m in ('XGB', 'GB'):                                   # exported training files were wrong; use saved models
                y = lg[(lg.Target == t) & (lg.Model == m)].iloc[0]
                d['Train_R2'] = y.fit_R2
        r5.append(dict(Target=t, Model=m, **d, p=pv.get(('LOSO', t, m)), n_perm=npm.get(('LOSO', t, m))))
    # mean predictor: mean of the other studies' records
    p = pr[(pr.Target == t) & (pr.Model == 'Ridge')]
    y = p.Actual.to_numpy(); s = p.Study_ID.to_numpy()
    mp = np.array([y[s != si].mean() for si in s])
    e = y - mp
    r5.append(dict(Target=t, Model='Mean predictor (= study-average baseline)', Test_R2=1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum(),
                   MAE=np.abs(e).mean(), RMSE=np.sqrt((e ** 2).mean())))
t5 = pd.DataFrame(r5)
studies = pr[pr.Model == 'Ridge'].groupby('Target').Study_ID.nunique().to_dict()
recs = pr[pr.Model == 'Ridge'].groupby('Target').size().to_dict()

t4.to_csv(os.path.join(HERE, 'Table4_KFold.csv'), index=False); t5.to_csv(os.path.join(HERE, 'Table5_LOSO.csv'), index=False)


# ---------------- formatted workbook
def fmtp(p):
    return '' if p is None or pd.isna(p) else f'{p:.3f}'


def sheet(ws, df, title, head, notes, extra):
    ws['A1'] = title; ws['A1'].font = Font(bold=True, name='Arial', size=11)
    cols = ['Target', 'Model', 'Training R²', 'Testing R²', 'Testing MAE', 'Testing RMSE', 'Permutation p (n)']
    for j, c in enumerate(cols, 1):
        cell = ws.cell(row=2, column=j, value=c); cell.font = Font(bold=True, name='Arial', size=10)
        cell.border = Border(top=Side(style='thin'), bottom=Side(style='thin')); cell.alignment = Alignment(horizontal='center')
    r = 3
    for t in T:
        sub = df[df.Target == t]
        models = sub[sub.Model.isin(M)]
        best = models.Test_R2.max()
        first = True
        for x in sub.itertuples():
            lab = f'{TN[t]}\n{extra(t)}' if first else ''
            first = False
            vals = [lab, x.Model, '' if pd.isna(x.Train_R2) else round(x.Train_R2, 3), round(x.Test_R2, 3), round(x.MAE, 2),
                    round(x.RMSE, 2), '' if pd.isna(x.p) else f'{fmtp(x.p)} ({int(x.n_perm)})']
            for j, v in enumerate(vals, 1):
                c = ws.cell(row=r, column=j, value=v); c.font = Font(name='Arial', size=10, italic=x.Model not in M,
                                                                       bold=(x.Model in M and x.Test_R2 == best))
                c.alignment = Alignment(horizontal='left' if j <= 2 else 'center', vertical='top', wrap_text=(j == 1))
            r += 1
        for j in range(1, 8):
            ws.cell(row=r - 1, column=j).border = Border(bottom=Side(style='thin'))
    r += 1
    for n in notes:
        c = ws.cell(row=r, column=1, value=n); c.font = Font(name='Arial', size=9, italic=True); r += 1
    for col, w in zip('ABCDEFG', [20, 34, 12, 12, 12, 13, 18]):
        ws.column_dimensions[col].width = w


out = os.path.join(HERE, 'Tables4_5_model_comparison.xlsx')
with pd.ExcelWriter(out) as w:
    pd.DataFrame().to_excel(w, sheet_name='Table 4'); pd.DataFrame().to_excel(w, sheet_name='Table 5')
    t4.to_excel(w, sheet_name='data_T4', index=False); t5.to_excel(w, sheet_name='data_T5', index=False)
wb = load_workbook(out)
kb2 = kb.set_index(['Target', 'Baseline'])
sheet(wb['Table 4'], t4, 'Table 4. Nested 5-fold cross-validation: training and pooled out-of-fold testing performance', None, [
    'Training metrics: records each model was fitted on (80% fitting subset of each outer training set). Testing metrics: pooled over the five held-out folds; MAE and RMSE in the units of the target.',
    'Bold: highest testing R² for the target. Permutation p = (1 + m)/(1 + n), target shuffled against the inputs and the full outer procedure repeated n times with the per-fold hyperparameters; the smallest attainable p is 1/(1 + n) (0.005 for n = 199, 0.010 for n = 99).',
    'Study-average baseline: ignores all inputs; predicts each held-out record as the mean of the training records from the same study (training mean if none).',
    'CODEfflu DNN permutation: 49 runs with the authors\' Keras code; all other permutations re-implemented in Python on the same folds.'],
    lambda t: f'{int(kb2.loc[(t, "Mean predictor")].N)} records; {kb2.loc[(t, "Mean predictor")].same_study_share:.0%} with same-study training record')
sheet(wb['Table 5'], t5, 'Table 5. Leave-one-study-out validation: training and pooled held-out-study performance', None, [
    'Each fold holds out all records of one study; testing metrics pooled over all folds. Training metrics: 80% fitting subset (Ridge, RF, SVR, GB, XGB) or all training records (DNN, trained without a validation subset).',
    'Bold: highest testing R² for the target. Permutation tests were run for models with positive testing R²; p as in Table 4.',
    'Under LOSO the study-average baseline equals the mean predictor (the held-out study has no training records). R² < 0: less accurate than predicting the mean of the other studies.'],
    lambda t: f'{recs[t]} records, {studies[t]} studies')
wb.save(out)
pd.set_option('display.width', 200)
print(t4.round(3).to_string(index=False)); print(t5.round(3).to_string(index=False))

