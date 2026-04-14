"""
Genetic Algorithm for Feature Selection
Runs on BALANCED data to avoid biased feature selection.
Writes real-time progress to status file for UI display.
"""
import numpy as np
import random
import json
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
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
    if hasattr(y, 'values'): y = y.values
    if hasattr(X, 'values'): X = X.values
    minority_idx = np.where(y == 1)[0]
    majority_idx = np.where(y == 0)[0]
    n_to_add = len(majority_idx) - len(minority_idx)
    if n_to_add <= 0:
        return X, y
    sampled = np.random.choice(minority_idx, size=n_to_add, replace=True)
    return np.vstack([X, X[sampled]]), np.concatenate([y, y[sampled]])


def run_genetic_algorithm(dataset_path=None, pop_size=30, generations=15,
                           max_features=20, status_path=None):
    print("GA: Starting Feature Selection on BALANCED data...")

    _write_status(status_path, {
        'stage': 'GA', 'stage_label': 'Genetic Algorithm — Feature Selection',
        'step': 'Loading & balancing data...', 'gen': 0, 'total_gens': generations,
        'best_f1': 0, 'features': 0, 'pso_iter': 0, 'total_pso': 0,
        'done': False, 'error': None
    })

    X_train, X_test, y_train, y_test, scaler, feature_names = run_preprocessing(dataset_path)
    num_features = X_train.shape[1]
    X_bal, y_bal = _manual_oversample(X_train, y_train)
    bankrupt_n = int(y_bal.sum())
    healthy_n  = int((y_bal == 0).sum())
    print(f"  GA balanced: {bankrupt_n} bankrupt vs {healthy_n} non-bankrupt")

    pop = [np.random.randint(0, 2, num_features) for _ in range(pop_size)]
    for chrom in pop:
        if chrom.sum() == 0:
            chrom[np.random.randint(0, num_features)] = 1

    best_score = 0
    best_chrom = pop[0]

    for gen in range(generations):
        scores = []
        for chrom in pop:
            idx = np.where(chrom == 1)[0]
            if len(idx) == 0:
                scores.append(0); continue
            model = LogisticRegression(max_iter=500, class_weight='balanced', solver='liblinear')
            score = cross_val_score(model, X_bal[:, idx], y_bal, cv=3, scoring='f1').mean()
            scores.append(score)

        gen_best_idx = int(np.argmax(scores))
        if scores[gen_best_idx] > best_score:
            best_score = scores[gen_best_idx]
            best_chrom = pop[gen_best_idx].copy()

        n_selected = int(best_chrom.sum())
        print(f"  GA Gen {gen+1}/{generations} | Best F1: {best_score:.4f} | Features: {n_selected}")
        _write_status(status_path, {
            'stage': 'GA', 'stage_label': 'Genetic Algorithm — Feature Selection',
            'step': f'Generation {gen+1}/{generations} — Best F1: {best_score:.4f} | {n_selected} features',
            'gen': gen+1, 'total_gens': generations,
            'best_f1': round(float(best_score), 4), 'features': n_selected,
            'pso_iter': 0, 'total_pso': 0, 'done': False, 'error': None
        })

        ranked   = np.argsort(scores)[::-1]
        elite    = [pop[i].copy() for i in ranked[:pop_size // 2]]
        next_gen = elite[:]
        while len(next_gen) < pop_size:
            p1, p2 = random.sample(elite[:max(2, len(elite) // 2)], 2)
            cp      = random.randint(1, num_features - 1)
            child   = np.concatenate([p1[:cp], p2[cp:]])
            for i in range(len(child)):
                if random.random() < 0.02:
                    child[i] = 1 - child[i]
            if child.sum() == 0:
                child[np.random.randint(0, num_features)] = 1
            next_gen.append(child)
        pop = next_gen

    selected_indices = np.where(best_chrom == 1)[0]
    if len(selected_indices) > max_features:
        selected_indices = selected_indices[:max_features]
    selected_names = [feature_names[i] for i in selected_indices]
    print(f"GA: Selected {len(selected_names)} features.")

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(os.path.join(base, 'models'), exist_ok=True)
    with open(os.path.join(base, 'models', 'selected_features.json'), 'w') as f:
        json.dump(selected_names, f, indent=4)

    return selected_indices, selected_names, feature_names


if __name__ == "__main__":
    run_genetic_algorithm()
