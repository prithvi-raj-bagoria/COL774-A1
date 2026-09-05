import sys
import pandas as pd
import numpy as np
import pickle
import os
import gc
import time
import warnings

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_predict, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import f_classif

# Suppress non-critical convergence and runtime warnings
warnings.filterwarnings('ignore')


def fast_threshold_search(probs, y, n_th=5000):
    """
    Vectorized search for the optimal decision threshold that maximizes the 
    assignment evaluation metric:
        M = TPR - 100 * FPR
    """
    # Count total true positive (P) and true negative (N) windows in the dataset
    P = (y == 1).sum()
    N = (y == 0).sum()
    
    # Generate fine-grained threshold candidates:
    # 1. Logarithmic grid near 1.0 (to search subtle decision boundaries)
    # 2. Linear grid between 0.30 and 0.90
    tail = 1.0 - np.logspace(-1, -5, n_th // 2)
    linear = np.linspace(0.30, 0.90, n_th // 2)
    thresholds = np.unique(np.sort(np.concatenate([linear, tail])))
    
    # Broadcast predictions across all candidate thresholds at once:
    # preds matrix shape: (N_samples, N_thresholds)
    preds = probs[:, None] >= thresholds[None, :]
    y_mask = y[:, None] == 1
    
    # Calculate True Positives (TP) and False Positives (FP) per threshold
    TP = (preds & y_mask).sum(axis=0)
    FP = (preds & ~y_mask).sum(axis=0)
    
    # Calculate True Positive Rate (TPR) and False Positive Rate (FPR)
    tpr = TP / P
    fpr = FP / N
    
    # M-score metric calculation
    M = tpr - 100.0 * fpr
    
    # Filter thresholds to enforce minimum valid detection criteria
    valid = tpr >= 0.11
    strict_valid = valid & (fpr <= 0.005)
    
    # Select best threshold candidate based on safety priority
    if np.any(strict_valid):
        best_idx = np.argmax(np.where(strict_valid, M, -np.inf))
    elif np.any(valid):
        best_idx = np.argmin(np.where(valid, fpr, np.inf))
    else:
        best_idx = np.argmax(tpr)
        
    return thresholds[best_idx], M[best_idx], tpr[best_idx], fpr[best_idx]


def main():
    if len(sys.argv) not in (4, 5):
        print("Usage: python3 part_c.py <dataset_dir> <model.pkl> <final_features.csv>")
        sys.exit(1)

    t_total_start = time.perf_counter()
    dataset_dir = sys.argv[1]
    model_path = sys.argv[2]
    features_csv = sys.argv[3]

    # =========================================================================
    # PHASE 1: DATA LOADING
    # =========================================================================
    print("Loading data...")
    train = pd.read_csv(os.path.join(dataset_dir, 'train.csv'))
    val = pd.read_csv(os.path.join(dataset_dir, 'val.csv'))
    test = pd.read_csv(os.path.join(dataset_dir, 'test.csv'))

    # Keep only valid binary labels (0 = Non-AF, 1 = AF)
    train = train[train['label'].isin([0, 1])]
    val = val[val['label'].isin([0, 1])]
    
    # Identify numerical feature columns
    drop_cols = ['release_id', 'patient', 'label']
    feat_cols = [c for c in train.columns if c not in drop_cols]

    # =========================================================================
    # PHASE 2: DYNAMIC PATIENT FILTERING (Pure & Net-Liability Toxic Patients)
    # =========================================================================
    print("Dynamically filtering pure and noisy patients...")
    
    # Step 2a: Identify "Pure" patients (patients with only 1 class label in train)
    # Pure patients must be dropped to prevent Leave-One-Group-Out (LOPO) CV 
    # from crashing when calculating fold-level metrics (e.g. division by zero P=0).
    patient_class_counts = train.groupby('patient')['label'].nunique()
    pure_patients = patient_class_counts[patient_class_counts == 1].index.tolist()
    
    # Create temporary dataset without pure patients to run diagnostic check
    temp_train = train[~train['patient'].isin(pure_patients)]
    X_tmp = temp_train[feat_cols].values.astype(np.float32)
    X_tmp[np.isinf(X_tmp)] = np.nan
    y_tmp = temp_train['label'].values.astype(np.int8)
    g_tmp = temp_train['patient'].values

    # Step 2b: Quick diagnostic baseline model to compute patient-level M-score contributions
    pipe_check = make_pipeline(
        SimpleImputer(strategy='median'),
        RobustScaler(),
        LogisticRegression(
            penalty='l1',
            solver='liblinear',
            class_weight={0: 5.0, 1: 1.0},
            C=0.05,
            random_state=42,
            max_iter=100
        )
    )
    
    # Predict out-of-fold probabilities for diagnostic filtering
    probs_tmp = cross_val_predict(
        pipe_check,
        X_tmp,
        y_tmp,
        groups=g_tmp,
        cv=LeaveOneGroupOut(),
        method='predict_proba',
        n_jobs=4
    )[:, 1]

    # Evaluate exact metric liability contribution per patient:
    # m_contrib = (TP / P_total) - 100 * (FP / N_total)
    P_total = (y_tmp == 1).sum()
    N_total = (y_tmp == 0).sum()

    preds_tmp = (probs_tmp >= 0.85).astype(int)
    patient_scores = []
    
    for p_id in np.unique(g_tmp):
        mask = (g_tmp == p_id)
        tp = (preds_tmp[mask] & y_tmp[mask]).sum()
        fp = (preds_tmp[mask] & ~y_tmp[mask]).sum()
        
        m_contrib = (tp / P_total) - 100.0 * (fp / N_total)
        patient_scores.append((p_id, m_contrib, fp))
        
    # Patients whose net M-score contribution is strictly negative (< 0) with >0 FPs
    toxic_patients = [
        p[0] for p in sorted(patient_scores, key=lambda x: x[1]) 
        if p[1] < 0 and p[2] > 0
    ]
    
    # Dynamic Safety Cap: Never drop more than 15% of available training patients
    max_drops = max(1, int(len(np.unique(g_tmp)) * 0.15))
    noisy_patients = toxic_patients[:max_drops]
    
    print(f"  -> Dropping pure patients: {pure_patients}")
    print(f"  -> Dropping toxic patients (Net-Negative M-Score): {noisy_patients}")
    
    # Exclude identified bad patients from training dataset
    bad_patients = set(pure_patients + noisy_patients)
    train = train[~train['patient'].isin(bad_patients)]

    # =========================================================================
    # PHASE 3: JOBLIB MULTIPROCESSING CLEANUP
    # =========================================================================
    # Explicitly release diagnostic memory and terminate Loky worker processes.
    # Prevents worker pool deadlock when initializing the second cross_val_predict pool.
    del X_tmp, y_tmp, probs_tmp, g_tmp, temp_train, pipe_check
    gc.collect()
    try:
        from joblib.externals.loky import get_reusable_executor
        get_reusable_executor().shutdown(wait=True)
    except Exception:
        pass
    gc.collect()

    # =========================================================================
    # PHASE 4: EXECUTION MODE SELECTION (Local EVAL vs. Kaggle Production)
    # =========================================================================
    if "temp_eval" in dataset_dir:
        # Local evaluation mode: Train on train.csv only
        all_labeled = train.copy()
        C_param = 0.01
        print("Running in EVAL mode (Train only) | C=0.01")
    else:
        # Kaggle submission mode: Concatenate Train + Val for 30% more data!
        all_labeled = pd.concat([train, val], ignore_index=True)
        C_param = 0.005
        print("Running in KAGGLE mode (Train + Val concat) | C=0.005")

    # Extract feature matrices and target arrays
    X_all = all_labeled[feat_cols].values.astype(np.float32)
    y_all = all_labeled['label'].values.astype(np.int8)
    g_all = all_labeled['patient'].values
    
    X_te = test[feat_cols].values.astype(np.float32)
    test_release_ids = test['release_id'].values

    # Clean infinite values
    X_all[np.isinf(X_all)] = np.nan
    X_te[np.isinf(X_te)] = np.nan

    # Load separate validation set for threshold validation in EVAL mode
    val_df = pd.read_csv(os.path.join(dataset_dir, 'val.csv'))
    val_df = val_df[val_df['label'].isin([0, 1])]
    X_v = val_df[feat_cols].values.astype(np.float32)
    y_v = val_df['label'].values.astype(np.int8)
    X_v[np.isinf(X_v)] = np.nan

    del train, val, all_labeled
    gc.collect()

    # =========================================================================
    # PHASE 5: FEATURE EXPANSION (Degree-2 Polynomial Interactions)
    # =========================================================================
    print("Expanding non-linear features (Degree 2 Polynomials)...")
    
    # Pre-scale features to select top features cleanly via ANOVA F-score
    base_imp = SimpleImputer(strategy='median')
    base_scl = RobustScaler()
    
    X_all_prep = base_scl.fit_transform(base_imp.fit_transform(X_all))
    X_te_prep = base_scl.transform(base_imp.transform(X_te))
    X_v_prep = base_scl.transform(base_imp.transform(X_v))
    
    # Select Top 10 features via ANOVA F-value ranking
    F_stats, _ = f_classif(X_all_prep, y_all)
    top_k_idx = np.argsort(np.nan_to_num(F_stats))[-10:]
    
    # Generate degree 2 interaction terms:
    # 10 linear terms produce 55 polynomial features (10 linear + 45 interaction terms).
    # Slicing [:, 10:] keeps only the 45 non-linear interaction terms.
    # Total features = 392 original + 45 interaction terms = 437 features (<500 limit).
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_all_poly = poly.fit_transform(X_all_prep[:, top_k_idx])[:, 10:]
    X_te_poly = poly.transform(X_te_prep[:, top_k_idx])[:, 10:]
    X_v_poly = poly.transform(X_v_prep[:, top_k_idx])[:, 10:]
    
    # Combine original features with new interaction features
    X_all = np.hstack([X_all, X_all_poly])
    X_te = np.hstack([X_te, X_te_poly])
    X_v = np.hstack([X_v, X_v_poly])
    print(f"✓ Feature dimension increased to {X_all.shape[1]} (Limit: <500)")

    # =========================================================================
    # PHASE 6: LOPO CROSS-VALIDATION & MODEL TRAINING
    # =========================================================================
    print(f"Running LOPO CV & fitting final model (C={C_param})...")
    
    # Main training pipeline: Impute -> RobustScale -> L1 Logistic Regression
    pipeline = make_pipeline(
        SimpleImputer(strategy='median'),
        RobustScaler(),
        LogisticRegression(
            penalty='l1',
            solver='liblinear',
            class_weight={0: 100.0, 1: 1.0},
            C=C_param,
            max_iter=150,
            random_state=42
        )
    )

    # Perform Leave-One-Group-Out (LOPO) cross-validation
    cv = LeaveOneGroupOut()
    train_probs = cross_val_predict(
        pipeline,
        X_all,
        y_all,
        groups=g_all,
        cv=cv,
        method='predict_proba',
        n_jobs=4
    )[:, 1]
    
    # Fit final pipeline on all available training data
    pipeline.fit(X_all, y_all)

    # =========================================================================
    # PHASE 7: THRESHOLD OPTIMIZATION
    # =========================================================================
    # Avoid double-counting validation predictions when running in Kaggle mode
    if "temp_eval" in dataset_dir:
        val_probs = pipeline.predict_proba(X_v)[:, 1]
        pooled_probs = np.concatenate([train_probs, val_probs])
        pooled_y = np.concatenate([y_all, y_v])
    else:
        # In Kaggle mode, X_all already includes Val set, so train_probs covers all samples
        pooled_probs = train_probs
        pooled_y = y_all
        
    # Search for global optimal probability threshold
    win_th, win_M, win_tpr, win_fpr = fast_threshold_search(pooled_probs, pooled_y)
    print(f"  Window-level Th: {win_th:.4f} | Pooled M: {win_M:.4f}")

    # Extract pipeline steps for inspection and export
    clf = pipeline.named_steps['logisticregression']
    imp = pipeline.named_steps['simpleimputer']
    scaler = pipeline.named_steps['robustscaler']
    
    print(f"Non-zero L1 features: {np.sum(clf.coef_[0] != 0)} / {X_all.shape[1]}")

    # =========================================================================
    # PHASE 8: SAVE MODEL DICTIONARY & SCALED TEST FEATURES
    # =========================================================================
    # Preprocess test set using fitted imputer and scaler
    X_te_imp = imp.transform(X_te)
    X_te_final = scaler.transform(X_te_imp)

    # Dump weights, bias, and threshold for evaluation scripts
    model_dict = {
        'weights': clf.coef_[0].astype(np.float64),
        'bias': float(clf.intercept_[0]),
        'threshold': float(win_th)
    }
    with open(model_path, 'wb') as f:
        pickle.dump(model_dict, f)

    # Write final preprocessed test features to CSV
    feat_col_names = [f"feat_{i}" for i in range(X_te_final.shape[1])]
    out_df = pd.DataFrame(X_te_final, columns=feat_col_names)
    out_df['release_id'] = test_release_ids
    out_df.to_csv(features_csv, index=False)


if __name__ == '__main__':
    main()
