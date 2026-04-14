"""
PSO Optimization — Option B Final
QuantileTransformer + Synthetic Extreme Samples + Balanced Training
Writes real-time progress to status file for UI display.
"""
import numpy as np
import joblib
import json
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import QuantileTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from algorithms.preprocessing import run_preprocessing


def _write_status(status_path, data):
    if status_path:
        try:
            with open(status_path, 'w') as f:
                json.dump(data, f)
        except Exception:
            pass


def _manual_oversample(X, y, random_state=42):
    np.random.seed(random_state)
    minority_idx = np.where(y == 1)[0]
    majority_idx = np.where(y == 0)[0]
    n_to_add = len(majority_idx) - len(minority_idx)
    if n_to_add <= 0:
        return X, y
    sampled = np.random.choice(minority_idx, size=n_to_add, replace=True)
    return np.vstack([X, X[sampled]]), np.concatenate([y, y[sampled]])


def _generate_synthetic_samples(feature_names, feature_means):
    try:
        from feature_engine import compute_all_features
    except ImportError:
        return None, None

    extreme_bankrupts = [
        {'revenue': 100000,  'total_assets': 500000,  'total_liabilities': 600000,
         'operating_income': -60000,  'net_income': -90000,  'op_cash_flow': -30000,
         'current_assets': 80000,  'current_liabilities': 200000},
        {'revenue': 200000,  'total_assets': 800000,  'total_liabilities': 900000,
         'operating_income': -80000,  'net_income': -120000, 'op_cash_flow': -40000,
         'current_assets': 100000, 'current_liabilities': 350000},
        {'revenue': 0,       'total_assets': 400000,  'total_liabilities': 500000,
         'operating_income': -100000, 'net_income': -150000, 'op_cash_flow': -60000,
         'current_assets': 50000,  'current_liabilities': 300000},
        {'revenue': 500000,  'total_assets': 600000,  'total_liabilities': 700000,
         'operating_income': -50000,  'net_income': -80000,  'op_cash_flow': -20000,
         'current_assets': 90000,  'current_liabilities': 250000},
        {'revenue': 300000,  'total_assets': 1000000, 'total_liabilities': 1100000,
         'operating_income': -150000, 'net_income': -200000, 'op_cash_flow': -80000,
         'current_assets': 120000, 'current_liabilities': 500000},
    ]
    extreme_healthy = [
        {'revenue': 5000000, 'total_assets': 3000000, 'total_liabilities': 500000,
         'operating_income': 1000000, 'net_income': 750000,  'op_cash_flow': 900000,
         'current_assets': 2000000, 'current_liabilities': 200000},
        {'revenue': 2000000, 'total_assets': 1000000, 'total_liabilities': 200000,
         'operating_income': 500000,  'net_income': 380000,  'op_cash_flow': 450000,
         'current_assets': 700000,  'current_liabilities': 80000},
        {'revenue': 800000,  'total_assets': 400000,  'total_liabilities': 60000,
         'operating_income': 160000,  'net_income': 120000,  'op_cash_flow': 150000,
         'current_assets': 300000,  'current_liabilities': 30000},
    ]

    def raw_to_row(raw):
        computed = compute_all_features(raw)
        row = []
        for col in feature_names:
            val = computed.get(col, feature_means.get(col, 0.0))
            if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
                val = feature_means.get(col, 0.0)
            row.append(float(val))
        return row

    X_b = np.array([raw_to_row(r) for r in extreme_bankrupts])
    X_h = np.array([raw_to_row(r) for r in extreme_healthy])
    X_synth = np.vstack([X_b, X_h])
    y_synth = np.array([1] * len(X_b) + [0] * len(X_h))
    return X_synth, y_synth


def run_pso(n_particles=10, iterations=8, dataset_path=None, status_path=None):
    print("PSO (Option B): QuantileTransform + Synthetic Samples Training...")
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    _write_status(status_path, {
        'stage': 'PSO', 'stage_label': 'PSO — Hyperparameter Optimization',
        'step': 'Preparing training data with synthetic samples...',
        'gen': 15, 'total_gens': 15,
        'best_f1': 0, 'features': 0,
        'pso_iter': 0, 'total_pso': iterations,
        'done': False, 'error': None
    })

    X_train, X_test, y_train, y_test, old_scaler, feature_names = run_preprocessing(dataset_path)
    X_train = np.array(X_train); X_test = np.array(X_test)
    y_train_arr = np.array(y_train); y_test_arr = np.array(y_test)

    X_train_orig = old_scaler.inverse_transform(X_train)
    X_test_orig  = old_scaler.inverse_transform(X_test)

    X_bal, y_bal = _manual_oversample(X_train_orig, y_train_arr)
    print(f"  Balanced: {y_bal.sum():.0f} bankrupt vs {(y_bal==0).sum()} non-bankrupt")

    means_path = os.path.join(base, 'models', 'feature_means.pkl')
    feature_means = joblib.load(means_path) if os.path.exists(means_path) else {}
    feature_means.pop('Bankrupt?', None)

    X_synth, y_synth = _generate_synthetic_samples(feature_names, feature_means)
    if X_synth is not None:
        WEIGHT = 200
        X_synth_w = np.tile(X_synth, (WEIGHT, 1))
        y_synth_w = np.tile(y_synth, WEIGHT)
        X_combined = np.vstack([X_bal, X_synth_w])
        y_combined  = np.concatenate([y_bal, y_synth_w])
        print(f"  +Synthetic samples (weight={WEIGHT}): total {len(y_combined)} samples")
    else:
        X_combined, y_combined = X_bal, y_bal

    _write_status(status_path, {
        'stage': 'PSO', 'stage_label': 'PSO — Hyperparameter Optimization',
        'step': f'Data ready ({len(y_combined):,} samples). Fitting QuantileTransformer...',
        'gen': 15, 'total_gens': 15,
        'best_f1': 0, 'features': 0,
        'pso_iter': 0, 'total_pso': iterations,
        'done': False, 'error': None
    })

    qt = QuantileTransformer(output_distribution='uniform', random_state=42, n_quantiles=500)
    X_qt = qt.fit_transform(X_combined)
    X_test_qt = qt.transform(X_test_orig)

    selector = SelectKBest(f_classif, k=min(20, X_qt.shape[1]))
    selector.fit(X_qt, y_combined)
    selected_idx   = np.where(selector.get_support())[0]
    selected_names = [feature_names[i] for i in selected_idx]
    print(f"  Selected {len(selected_idx)} features")

    X_sel_train = X_qt[:, selected_idx]
    X_sel_test  = X_test_qt[:, selected_idx]

    # PSO for LR C parameter
    positions  = np.random.uniform(-2, 2, (n_particles, 1))
    velocities = np.random.uniform(-0.3, 0.3, (n_particles, 1))
    p_best_pos = positions.copy(); p_best_val = np.full(n_particles, -np.inf)
    g_best_pos = positions[0].copy(); g_best_val = -np.inf
    w, c1, c2 = 0.7, 1.5, 1.5

    for it in range(iterations):
        for i in range(n_particles):
            C_val = 10 ** float(positions[i, 0])
            m     = LogisticRegression(C=C_val, max_iter=300, class_weight='balanced', solver='liblinear')
            score = cross_val_score(m, X_sel_train, y_combined, cv=3, scoring='f1').mean()
            if score > p_best_val[i]: p_best_val[i] = score; p_best_pos[i] = positions[i].copy()
            if score > g_best_val:    g_best_val = score;     g_best_pos = positions[i].copy()
        r1, r2 = np.random.rand(), np.random.rand()
        velocities = w * velocities + c1 * r1 * (p_best_pos - positions) + c2 * r2 * (g_best_pos - positions)
        positions  = np.clip(positions + velocities, -2, 2)
        print(f"  PSO Iter {it+1}/{iterations} | Best F1: {g_best_val:.4f}")
        _write_status(status_path, {
            'stage': 'PSO', 'stage_label': 'PSO — Hyperparameter Optimization',
            'step': f'PSO Iteration {it+1}/{iterations} — Best F1: {g_best_val:.4f} | C={10**float(g_best_pos[0]):.4f}',
            'gen': 15, 'total_gens': 15,
            'best_f1': round(float(g_best_val), 4), 'features': len(selected_idx),
            'pso_iter': it + 1, 'total_pso': iterations,
            'done': False, 'error': None
        })

    best_C = float(10 ** g_best_pos[0])

    _write_status(status_path, {
        'stage': 'MODELS', 'stage_label': 'Training Final Models',
        'step': 'Training Logistic Regression + Random Forest...',
        'gen': 15, 'total_gens': 15, 'best_f1': round(float(g_best_val), 4),
        'features': len(selected_idx), 'pso_iter': iterations, 'total_pso': iterations,
        'done': False, 'error': None
    })

    model_lr = LogisticRegression(C=best_C, max_iter=1000, class_weight='balanced', solver='liblinear')
    model_lr.fit(X_sel_train, y_combined)
    score_lr = f1_score(y_test_arr, model_lr.predict(X_sel_test), zero_division=0)

    model_rf = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42, class_weight='balanced')
    model_rf.fit(X_sel_train, y_combined)
    score_rf = f1_score(y_test_arr, model_rf.predict(X_sel_test), zero_division=0)

    print(f"  LR F1: {score_lr:.4f} | RF F1: {score_rf:.4f}")
    best_model = model_rf if score_rf >= score_lr else model_lr
    algo_name  = "Random Forest" if score_rf >= score_lr else "Logistic Regression"

    y_probs = best_model.predict_proba(X_sel_test)[:, 1]
    best_thresh, best_f1 = 0.50, 0.0
    for thresh in np.arange(0.05, 0.90, 0.05):
        yp = (y_probs >= thresh).astype(int)
        f1 = f1_score(y_test_arr, yp, zero_division=0)
        if f1 > best_f1: best_f1 = f1; best_thresh = thresh
    print(f"  Optimal threshold: {best_thresh:.2f} (F1={best_f1:.3f})")

    def get_metrics(model, X, y, thresh):
        probs = model.predict_proba(X)[:, 1]
        pred  = (probs >= thresh).astype(int)
        return {
            'accuracy':  round(float(accuracy_score(y, pred)) * 100, 2),
            'precision': round(float(precision_score(y, pred, zero_division=0)) * 100, 2),
            'recall':    round(float(recall_score(y, pred, zero_division=0)) * 100, 2),
            'f1':        round(float(f1_score(y, pred, zero_division=0)) * 100, 2),
        }
    metrics    = get_metrics(best_model, X_sel_test, y_test_arr, best_thresh)
    metrics_lr = get_metrics(model_lr,   X_sel_test, y_test_arr, best_thresh)
    metrics_rf = get_metrics(model_rf,   X_sel_test, y_test_arr, best_thresh)

    models_dir = os.path.join(base, 'models')
    os.makedirs(models_dir, exist_ok=True)

    import pandas as pd
    df_raw = pd.read_csv(dataset_path or os.path.join(base, 'dataset', 'bankruptcy.csv'))
    df_raw.columns = df_raw.columns.str.strip()
    from algorithms.preprocessing import detect_target_column
    target_col = detect_target_column(df_raw)
    df_features_raw = df_raw.drop(target_col, axis=1).select_dtypes(include=[np.number])
    feature_means_new = df_features_raw.mean().to_dict()
    joblib.dump(feature_means_new, os.path.join(models_dir, 'feature_means.pkl'))

    joblib.dump(model_lr, os.path.join(models_dir, 'logistic_model.pkl'))
    joblib.dump(model_rf, os.path.join(models_dir, 'random_forest_model.pkl'))

    bundle = {
        'model': best_model, 'scaler': qt, 'scaler_type': 'quantile',
        'selected_idx': selected_idx, 'selected_names': selected_names,
        'dataset_columns': feature_names, 'algo_name': algo_name,
        'metrics': metrics, 'metrics_lr': metrics_lr, 'metrics_rf': metrics_rf,
        'best_C': best_C, 'optimal_threshold': float(best_thresh),
    }
    joblib.dump(bundle, os.path.join(models_dir, 'best_model.pkl'))

    model_comparison = {
        'winner': algo_name, 'optimal_threshold': float(best_thresh),
        'logistic_regression': metrics_lr, 'random_forest': metrics_rf,
    }
    with open(os.path.join(models_dir, 'model_comparison.json'), 'w') as f:
        json.dump(model_comparison, f, indent=4)

    with open(os.path.join(models_dir, 'model_info.txt'), 'w') as f:
        f.write(f"Best Model: {algo_name}\nAccuracy: {metrics['accuracy']}%\n"
                f"F1 Score: {metrics['f1']}%\nOptimal Threshold: {best_thresh}\n"
                f"Scaler: QuantileTransformer\n")

    print(f"All models saved. Winner: {algo_name}")
    return algo_name, metrics


if __name__ == "__main__":
    run_pso()
