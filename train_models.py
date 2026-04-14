"""
Training Orchestrator — runs GA then PSO, passes status_path for live progress.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASE       = os.path.dirname(os.path.abspath(__file__))
STATUS_FILE = os.path.join(BASE, 'models', 'training_status.json')

def write_status(data):
    os.makedirs(os.path.join(BASE, 'models'), exist_ok=True)
    with open(STATUS_FILE, 'w') as f:
        json.dump(data, f)

def train(dataset_path=None):
    try:
        write_status({
            'stage': 'START', 'stage_label': 'Initializing...',
            'step': 'Starting training pipeline...', 'gen': 0, 'total_gens': 15,
            'best_f1': 0, 'features': 0, 'pso_iter': 0, 'total_pso': 8,
            'done': False, 'error': None
        })

        from algorithms.genetic_algorithm import run_genetic_algorithm
        selected_indices, selected_names, feature_names = run_genetic_algorithm(
            dataset_path=dataset_path,
            status_path=STATUS_FILE
        )

        from algorithms.pso_optimization import run_pso
        algo, metrics = run_pso(
            dataset_path=dataset_path,
            status_path=STATUS_FILE
        )

        write_status({
            'stage': 'DONE', 'stage_label': 'Training Complete!',
            'step': f'Best model: {algo} | Accuracy: {metrics["accuracy"]}% | F1: {metrics["f1"]}%',
            'gen': 15, 'total_gens': 15,
            'best_f1': metrics["f1"] / 100, 'features': len(selected_names),
            'pso_iter': 8, 'total_pso': 8,
            'done': True, 'error': None,
            'algo': algo, 'metrics': metrics
        })

        return algo, metrics

    except Exception as e:
        import traceback
        err = traceback.format_exc()
        write_status({
            'stage': 'ERROR', 'stage_label': 'Training Failed',
            'step': str(e), 'gen': 0, 'total_gens': 15,
            'best_f1': 0, 'features': 0, 'pso_iter': 0, 'total_pso': 8,
            'done': True, 'error': str(e)
        })
        print(err)
        return None, None


if __name__ == "__main__":
    algo, metrics = train()
    if algo:
        print(f"Winner: {algo}, Accuracy: {metrics['accuracy']}%")
