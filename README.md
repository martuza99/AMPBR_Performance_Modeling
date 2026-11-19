# Machine learning prediction of AMPBR performance from compiled literature data — code and data

Code, data and validation settings for: *Machine Learning Prediction of Algal Membrane Photobioreactor Performance from
Compiled Literature Data: Leakage-Free, Study-Aware Validation* (Desalination and Water Treatment, DWT-D-26-01475).

## Data (`data/`)
| File | Content |
|---|---|
| `Table_S1_dataset_with_study_IDs.xlsx` | The 69 operating conditions from 29 studies (Supplementary Table S1); column `Ref` = source study |
| `AMPBR_records_raw_with_missing.csv` | The same records as used for modelling, missing inputs left empty; `Study_ID` = source study |

* Inputs (10): influent COD, N and P; pH; temperature; flux; HRT; SRT; light intensity; photoperiod.
* Targets (4, modelled separately): biomass production (BMP, 57 records/22 studies), effluent COD (45/14),
  effluent N (68/28), effluent P (68/28). Each target uses the records on which it is reported.
* Unreported reactor temperature = 23 °C (ambient; applied before any split). All other missing inputs are mean-imputed
  with the training records of each outer fold only.

## Validation settings
| Setting | Value |
|---|---|
| Record-level validation | Nested 5-fold CV; folds stratified by the target (10 bins of ascending target, records allocated to the 5 folds in turn); `splits/fold_assignment_5fold_*.csv` |
| Study-level validation | Leave-one-study-out (LOSO); `splits/LOSO_fold_to_study.csv` |
| Preprocessing | Mean imputation, standardisation of inputs and target: fitted on training data of each fold only |
| Hyperparameter tuning | Inside every outer fold, training data only (manuscript Table 1 search spaces). Reported runs: random 80/20 fit/validation split; check with study-grouped inner validation (GroupKFold over training studies): `scripts/loso_grouped_inner.py` |
| Models | Ridge, RF, SVR (RBF), gradient boosting, XGBoost, DNN (2 or 4 hidden layers, 8–128 units, ReLU, Adam, learning rate 0.01) |
| Metrics | R², MAE, RMSE on pooled out-of-fold predictions, physical units |
| Baselines | Mean of training records; study-average of the training records of the same study |
| Permutation tests | Target shuffled across records (and, in the check, study means shuffled across studies); whole outer procedure repeated; `scripts/round2_fit80.py`, `scripts/perm_r3.py` |
| CSI comparison | Interpolated records added to the training folds only; `scripts/csi_compare_tuned.py` |
| Sensitivity | Sobol (Saltelli), effluent P, five 5-fold XGB models; `scripts/sobol_PR.py` |
| Random seeds | Fixed in every script (see headers) |

## Code
* `notebooks/` — the notebooks that produced the reported models (outputs stripped): DNN, XGB/GB and Ridge/SVR/RF for
  5-fold CV and LOSO. Paths at the top of each notebook point to the authors' folders and must be adapted.
* `scripts/` — validation, baselines, permutation tests, CSI comparison, Sobol analysis and the round-3 checks
  (study-grouped inner validation, re-tuned permutation tests, BMP extreme-record sensitivity).

Environment used: Python 3.11, scikit-learn 1.9, xgboost 3.2, bayes_opt 3.3, TensorFlow/Keras 2.x with Keras Tuner, SALib.

## Licence
Code: MIT. Data: CC BY 4.0 (compiled from the published studies cited in Table S1).
