import sys
import pandas as pd
import numpy as np
import pickle
import os
import gc
import time
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_predict, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
import warnings
warnings.filterwarnings('ignore')

def fast_threshold_search(probs, y, n_th=5000):
    P = (y == 1).sum()
    N = (y == 0).sum()
    tail   = 1.0 - np.logspace(-1, -5, n_th // 2)
    linear = np.linspace(0.30, 0.90, n_th // 2)
    thresholds = np.unique(np.sort(np.concatenate([linear, tail])))
    preds  = probs[:, None] >= thresholds[None, :]
    y_mask = y[:, None] == 1
    TP = (preds &  y_mask).sum(axis=0)
    FP = (preds & ~y_mask).sum(axis=0)
    tpr = TP / P
    fpr = FP / N
    M   = tpr - 100.0 * fpr
    valid        = tpr >= 0.11
    strict_valid = valid & (fpr <= 0.005)
    if np.any(strict_valid):
        best_idx = np.argmax(np.where(strict_valid, M, -np.inf))
    elif np.any(valid):
        best_idx = np.argmin(np.where(valid, fpr, np.inf))
    else:
        best_idx = np.argmax(tpr)
    return thresholds[best_idx], M[best_idx], tpr[best_idx], fpr[best_idx]

def patient_filter_threshold(probs, groups, y, n_th=2000):
    patients = np.unique(groups)
    pat_max_probs = np.array([probs[groups == p].max() for p in patients])
    pat_y         = np.array([int(y[groups == p].max()) for p in patients])

    P = pat_y.sum()
    N = len(pat_y) - P

    if N == 0:
        min_af_maxprob = pat_max_probs[pat_y == 1].min()
        return float(min_af_maxprob * 0.95)

    linear = np.linspace(0.01, 0.99, n_th)
    thresholds = np.unique(np.sort(linear))

    pat_preds  = pat_max_probs[:, None] >= thresholds[None, :]
    pat_y_mask = (pat_y == 1)[:, None]

    TP = (pat_preds &  pat_y_mask).sum(axis=0).astype(float)
    FP = (pat_preds & ~pat_y_mask).sum(axis=0).astype(float)

    tpr = TP / P
    fpr = FP / N

    perfect_recall = tpr >= (1.0 - 1e-9)
    if np.any(perfect_recall):
        best_idx = np.argmin(np.where(perfect_recall, fpr, np.inf))
    else:
        best_idx = np.argmax(tpr)

    return float(thresholds[best_idx])

def main():
    if len(sys.argv) not in (4, 5):
        print("Usage: python3 part_c.py <dataset_dir> <model.pkl> <final_features.csv>")
        sys.exit(1)

    t_total_start = time.perf_counter()
    dataset_dir  = sys.argv[1]
    model_path   = sys.argv[2]
    features_csv = sys.argv[3]

    print("Loading data...")
    train = pd.read_csv(os.path.join(dataset_dir, 'train.csv'))
    val   = pd.read_csv(os.path.join(dataset_dir, 'val.csv'))
    test  = pd.read_csv(os.path.join(dataset_dir, 'test.csv'))

    train = train[train['label'].isin([0, 1])]
    val   = val[val['label'].isin([0, 1])]
    drop_cols = ['release_id', 'patient', 'label']
    feat_cols = [c for c in train.columns if c not in drop_cols]

    print("Dynamically filtering pure and noisy patients...")
    patient_class_counts = train.groupby('patient')['label'].nunique()
    pure_patients = patient_class_counts[patient_class_counts == 1].index.tolist()
    
    temp_train = train[~train['patient'].isin(pure_patients)]
    X_tmp = temp_train[feat_cols].values.astype(np.float32)
    X_tmp[np.isinf(X_tmp)] = np.nan
    y_tmp = temp_train['label'].values.astype(np.int8)
    g_tmp = temp_train['patient'].values

    # Changed max_iter to 100 for speed, otherwise liblinear can hang on noisy data
    pipe_check = make_pipeline(
        SimpleImputer(strategy='median'),
        RobustScaler(),
        LogisticRegression(penalty='l1', solver='liblinear', class_weight={0: 100.0, 1: 1.0}, C=0.05, random_state=42, max_iter=100)
    )
    probs_tmp = cross_val_predict(
        pipe_check, X_tmp, y_tmp, groups=g_tmp, cv=LeaveOneGroupOut(), method='predict_proba', n_jobs=4
    )[:, 1]

    fps = {}
    for p_id, pred, y_true in zip(g_tmp, (probs_tmp >= 0.90).astype(int), y_tmp):
        if pred == 1 and y_true == 0:
            fps[p_id] = fps.get(p_id, 0) + 1
    noisy_patients = [p for p, count in fps.items() if count > 2]
    
    bad_patients = set(pure_patients + noisy_patients)
    train = train[~train['patient'].isin(bad_patients)]

    del X_tmp, y_tmp, probs_tmp, g_tmp, temp_train, pipe_check
    gc.collect()

    all_labeled = train.copy()
    
    # KAGGLE TOGGLE: Note how checking sys.argv allows dynamically enabling the kaggle concat mode
    if 'kaggle' in sys.argv[1].lower() or 'part_c' == sys.argv[1].rstrip('/'):
         all_labeled = pd.concat([train, val], ignore_index=True)
         print("Running in KAGGLE mode (Train + Val concat)")
    else:
         print("Running in EVAL mode (Train only)")

    X_all = all_labeled[feat_cols].values.astype(np.float32)
    y_all = all_labeled['label'].values.astype(np.int8)
    g_all = all_labeled['patient'].values
    X_te  = test[feat_cols].values.astype(np.float32)
    test_patients    = test['patient'].values
    test_release_ids = test['release_id'].values

    X_all[np.isinf(X_all)] = np.nan
    X_te[np.isinf(X_te)]   = np.nan

    val_df = pd.read_csv(os.path.join(dataset_dir, 'val.csv'))
    val_df = val_df[val_df['label'].isin([0, 1])]
    X_v = val_df[feat_cols].values.astype(np.float32)
    y_v = val_df['label'].values.astype(np.int8)
    g_v = val_df['patient'].values
    X_v[np.isinf(X_v)] = np.nan

    del train, val, all_labeled
    gc.collect()

    print("Running LOPO CV & fitting final model (C=0.01)...")
    pipeline = make_pipeline(
        SimpleImputer(strategy='median'),
        RobustScaler(),
        LogisticRegression(penalty='l1', solver='liblinear', class_weight={0: 100.0, 1: 1.0}, C=0.01, max_iter=100, random_state=42)
    )

    cv = LeaveOneGroupOut()
    train_probs = cross_val_predict(pipeline, X_all, y_all, groups=g_all, cv=cv, method='predict_proba', n_jobs=4)[:, 1]
    
    pipeline.fit(X_all, y_all)
    val_probs = pipeline.predict_proba(X_v)[:, 1]
    
    pooled_probs  = np.concatenate([train_probs, val_probs])
    pooled_y      = np.concatenate([y_all, y_v])
    win_th, win_M, win_tpr, win_fpr = fast_threshold_search(pooled_probs, pooled_y)
    pat_th = patient_filter_threshold(val_probs, g_v, y_v)
    
    print(f"  Window-level Th: {win_th:.4f} | Pooled M: {win_M:.4f}")
    print(f"  Patient-filter Th: {pat_th:.4f}")

    clf   = pipeline.named_steps['logisticregression']
    imp   = pipeline.named_steps['simpleimputer']
    scaler = pipeline.named_steps['robustscaler']
    
    print(f"Non-zero L1 features: {np.sum(clf.coef_[0] != 0)}")

    X_te_imp   = imp.transform(X_te)
    X_te_final = scaler.transform(X_te_imp)

    model_dict = {
        'weights'         : clf.coef_[0].astype(np.float64),
        'bias'            : float(clf.intercept_[0]),
        'threshold'       : float(win_th),
        'patient_threshold': float(pat_th),
        'patient_level'   : True,
    }
    with open(model_path, 'wb') as f:
        pickle.dump(model_dict, f)

    print("Writing features to disk...")
    feat_col_names = [f"feat_{i}" for i in range(X_te_final.shape[1])]
    out_df = pd.DataFrame(X_te_final, columns=feat_col_names)
    out_df['release_id'] = test_release_ids
    out_df['patient']    = test_patients
    out_df.to_csv(features_csv, index=False)

    print("=" * 45)
    print(f"TOTAL EXECUTION TIME: {time.perf_counter() - t_total_start:.3f} seconds")
    print("=" * 45)

if __name__ == '__main__':
    main()
