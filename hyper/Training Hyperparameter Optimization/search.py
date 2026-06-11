import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    auc,
    matthews_corrcoef
)
import numpy as np
import os
import random
import logging
from datetime import datetime
from torch_geometric.data import Batch
import optuna
from optuna.samplers import TPESampler
import json

from model_windowsize import PLMGraphModel
from utils import load_fasta

WINDOW_SIZE = 41  
DATA_DIR = 'data'
K_FOLDS = 5
PATIENCE = 5  
EARLY_STOP_METRIC = 'MCC'
EARLY_STOP_MIN_DELTA = 1e-4
SEED = 0

N_TRIALS = 20  
TIMEOUT = None  

OUTPUT_DIR = 'bayesian_optimization'
os.makedirs(OUTPUT_DIR, exist_ok=True)

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f'{OUTPUT_DIR}/bayesian_opt_{current_time}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_ORDER)}
X_IDX = len(AA_ORDER)
SEQ_CENTER_IDX = WINDOW_SIZE // 2
DATA_TAG = f"{WINDOW_SIZE}aa"

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def extract_protein_name(fasta_id):
    parts = fasta_id.split('_')
    if len(parts) >= 2:
        return parts[1]
    return fasta_id


class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, graph_dict, seq_dict):
        self.ids = []
        self.graphs = []
        self.seq_onehots = []

        missing_seq = 0
        for sid, graph in graph_dict.items():
            seq = seq_dict.get(sid)
            if seq is None:
                missing_seq += 1
                continue

            self.ids.append(sid)
            self.graphs.append(graph)
            self.seq_onehots.append(seq_to_onehot(seq))

        if missing_seq > 0:
            logger.warning(f"Skipped {missing_seq} graph samples without matched FASTA sequence.")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return self.graphs[idx], self.seq_onehots[idx]


def seq_to_onehot(seq):
    seq = (seq or '').upper()
    onehot = torch.zeros((len(seq), 21), dtype=torch.float32)
    for i, aa in enumerate(seq):
        aa_idx = AA_TO_IDX.get(aa, X_IDX)
        onehot[i, aa_idx] = 1.0
    return onehot


def graph_collate(batch):
    graphs, seq_onehots = zip(*batch)
    graph_batch = Batch.from_data_list(list(graphs))

    max_len = max(seq.size(0) for seq in seq_onehots)
    seq_batch = torch.zeros((len(seq_onehots), max_len, 21), dtype=torch.float32)
    for i, seq in enumerate(seq_onehots):
        seq_batch[i, :seq.size(0), :] = seq

    return graph_batch, seq_batch


def calculate_metrics(y_true, y_pred, y_prob):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity + 1e-8)

    mcc = matthews_corrcoef(y_true, y_pred)

    roc_auc = roc_auc_score(y_true, y_prob)
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall_curve, precision_curve)

    bacc = (sensitivity + specificity) / 2

    return {
        'MCC': mcc,
        'SN': sensitivity,
        'SP': specificity,
        'ACC': accuracy,
        'BACC': bacc,
        'PRECISION': precision,
        'F1': f1,
        'AUC': roc_auc,
        'AUCPR': pr_auc
    }


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for graph_batch, seq_onehot in loader:
        graph_batch = graph_batch.to(device)
        seq_onehot = seq_onehot.to(device)
        optimizer.zero_grad()

        logits = model(graph_batch, seq_onehot)
        targets = graph_batch.y.view(-1)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate_model(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_probs, all_preds, all_labels = [], [], []

    with torch.no_grad():
        for graph_batch, seq_onehot in loader:
            graph_batch = graph_batch.to(device)
            seq_onehot = seq_onehot.to(device)

            logits = model(graph_batch, seq_onehot)
            targets = graph_batch.y.view(-1)
            loss = criterion(logits, targets)
            total_loss += loss.item()

            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(targets.cpu().numpy().tolist())

    metrics = calculate_metrics(all_labels, all_preds, all_probs)
    metrics['Val Loss'] = total_loss / len(loader)
    return metrics


def resolve_data_path(data_dir, candidates):
    for name in candidates:
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            return path
    return os.path.join(data_dir, candidates[0])


def validate_sequence_lengths(ids, seqs, expected_len, source_name):
    invalid = [(sid, len(seq)) for sid, seq in zip(ids, seqs) if len(seq) != expected_len]
    if invalid:
        preview = ', '.join([f"{sid}:{seq_len}" for sid, seq_len in invalid[:5]])
        raise ValueError(
            f"{source_name} has {len(invalid)} sequences not equal to {expected_len}. "
            f"Examples: {preview}"
        )


def load_data():
    train_graph_path = resolve_data_path(
        DATA_DIR,
        [f'rice_train_embedding_graph.pt']
    )
    test_graph_path = resolve_data_path(
        DATA_DIR,
        [f'rice_test_embedding_graph.pt']
    )

    logger.info(f"Loading training data from {train_graph_path}...")
    train_graph_dict = torch.load(train_graph_path)
    test_graph_dict = torch.load(test_graph_path)

    train_fasta = resolve_data_path(DATA_DIR, [f'train.fasta'])
    test_fasta = resolve_data_path(DATA_DIR, [f'test.fasta'])
    train_ids, train_seqs, _ = load_fasta(train_fasta)
    test_ids, test_seqs, _ = load_fasta(test_fasta)
    validate_sequence_lengths(train_ids, train_seqs, WINDOW_SIZE, train_fasta)
    validate_sequence_lengths(test_ids, test_seqs, WINDOW_SIZE, test_fasta)
    train_seq_dict = {sid: seq for sid, seq in zip(train_ids, train_seqs)}
    test_seq_dict = {sid: seq for sid, seq in zip(test_ids, test_seqs)}

    full_train_dataset = GraphDataset(train_graph_dict, train_seq_dict)

    if len(full_train_dataset) == 0:
        raise ValueError("No training samples after matching graph data with FASTA sequences.")

    groups = np.array([extract_protein_name(sid) for sid in full_train_dataset.ids])
    y_for_cv = np.array([graph.y.item() for graph in full_train_dataset.graphs])

    logger.info(f"Dataset loaded: {len(full_train_dataset)} samples")
    logger.info(f"Number of unique proteins (groups): {len(np.unique(groups))}")

    return full_train_dataset, groups, y_for_cv


def run_cross_validation(batch_size, lr, weight_decay, n_epochs,
                         full_train_dataset, groups, y_for_cv, device):
    criterion = nn.CrossEntropyLoss()
    group_kfold = GroupKFold(n_splits=K_FOLDS)
    fold_metrics = []
    fold_best_epochs = []

    x_dummy = np.zeros(len(y_for_cv), dtype=np.int64)

    for fold, (train_idx, val_idx) in enumerate(group_kfold.split(x_dummy, y_for_cv, groups)):
        train_proteins = set(groups[train_idx])
        val_proteins = set(groups[val_idx])
        overlap = train_proteins & val_proteins
        if overlap:
            logger.warning(f"Fold {fold+1}: Protein overlap detected! {overlap}")

        train_loader = DataLoader(
            Subset(full_train_dataset, train_idx.tolist()),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=graph_collate
        )
        val_loader = DataLoader(
            Subset(full_train_dataset, val_idx.tolist()),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=graph_collate
        )

        model = PLMGraphModel(
            node_dim=1280,
            edge_dim=16,
            hidden_dim=128,
            num_classes=2,
            seq_len=WINDOW_SIZE,
            seq_center_idx=SEQ_CENTER_IDX
        ).to(device)

        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        best_val_score = -float('inf')
        best_epoch = 0
        best_state_dict = None
        patience_counter = 0

        for epoch in range(n_epochs):
            train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_res = evaluate_model(model, val_loader, criterion, device)

            current_val_score = float(val_res[EARLY_STOP_METRIC])

            if current_val_score > best_val_score + EARLY_STOP_MIN_DELTA:
                best_val_score = current_val_score
                best_epoch = epoch + 1
                best_state_dict = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= PATIENCE:
                break

        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)
            final_metrics = evaluate_model(model, val_loader, criterion, device)
            fold_metrics.append(final_metrics[EARLY_STOP_METRIC])
        else:
            fold_metrics.append(best_val_score)

        fold_best_epochs.append(best_epoch)

    mean_mcc = np.mean(fold_metrics)
    std_mcc = np.std(fold_metrics)
    median_epoch = int(np.median(fold_best_epochs))
    return mean_mcc, std_mcc, fold_metrics, fold_best_epochs, median_epoch


def objective(trial, full_train_dataset, groups, y_for_cv, device):
    lr = trial.suggest_float('learning_rate', 1e-6, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256, 512])
    
    n_epochs = 200

    logger.info(f"\n{'='*60}")
    logger.info(f"Trial {trial.number + 1}")
    logger.info(f"{'='*60}")
    logger.info(f"Hyperparameters:")
    logger.info(f"  learning_rate: {lr:.6e}")
    logger.info(f"  weight_decay: {weight_decay:.6e}")
    logger.info(f"  batch_size: {batch_size}")
    logger.info(f"  epochs: {n_epochs}")

    try:
        mean_mcc, std_mcc, fold_metrics, fold_best_epochs, median_epoch = run_cross_validation(
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            n_epochs=n_epochs,
            full_train_dataset=full_train_dataset,
            groups=groups,
            y_for_cv=y_for_cv,
            device=device
        )

        logger.info(f"Cross-Validation Results:")
        logger.info(f"  Mean MCC: {mean_mcc:.4f} ± {std_mcc:.4f}")
        logger.info(f"  Fold MCCs: {[f'{m:.4f}' for m in fold_metrics]}")
        logger.info(f"  Fold Best Epochs: {fold_best_epochs}")
        logger.info(f"  Recommended Final Epochs (median): {median_epoch}")

        trial.set_user_attr('median_epoch', median_epoch)
        trial.set_user_attr('fold_best_epochs', fold_best_epochs)
        trial.set_user_attr('std_mcc', std_mcc)

        return mean_mcc

    except Exception as e:
        logger.error(f"Trial {trial.number + 1} failed with error: {str(e)}")
        return float('-inf')  


def run_bayesian_optimization():
    logger.info("=" * 70)
    logger.info("Bayesian Optimization for PLMGraphModel Hyperparameters")
    logger.info("=" * 70)
    logger.info(f"Window Size: {WINDOW_SIZE}")
    logger.info(f"K-Folds: {K_FOLDS}")
    logger.info(f"Early Stop Patience: {PATIENCE}")
    logger.info(f"Early Stop Metric: {EARLY_STOP_METRIC}")
    logger.info(f"Number of Trials: {N_TRIALS}")
    logger.info(f"Seed: {SEED}")
    logger.info("=" * 70)

    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    full_train_dataset, groups, y_for_cv = load_data()

    sampler = TPESampler(seed=SEED)
    study = optuna.create_study(
        direction='maximize', 
        sampler=sampler,
        study_name='plm_graph_model_optimization',
        storage=None,  
        load_if_exists=False
    )

    def objective_wrapper(trial):
        return objective(trial, full_train_dataset, groups, y_for_cv, device)

    logger.info("\nStarting Bayesian Optimization...")
    study.optimize(
        objective_wrapper,
        n_trials=N_TRIALS,
        timeout=TIMEOUT,
        show_progress_bar=True
    )

    logger.info("\n" + "=" * 70)
    logger.info("OPTIMIZATION COMPLETED")
    logger.info("=" * 70)

    logger.info(f"\nNumber of finished trials: {len(study.trials)}")
    logger.info(f"Best trial: {study.best_trial.number + 1}")
    logger.info(f"Best MCC: {study.best_value:.4f}")
    logger.info("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.6e}")
        else:
            logger.info(f"  {key}: {value}")

    best_median_epoch = study.best_trial.user_attrs.get('median_epoch', 200)
    best_fold_epochs = study.best_trial.user_attrs.get('fold_best_epochs', [])
    logger.info(f"\nRecommended epochs for final training:")
    logger.info(f"  Median of best epochs across folds: {best_median_epoch}")
    logger.info(f"  Best epochs per fold: {best_fold_epochs}")

    results = {
        'best_params': study.best_params,
        'best_mcc': float(study.best_value),
        'best_std_mcc': study.best_trial.user_attrs.get('std_mcc', None),
        'recommended_epochs': best_median_epoch,
        'fold_best_epochs': best_fold_epochs,
        'n_trials': len(study.trials),
        'window_size': WINDOW_SIZE,
        'k_folds': K_FOLDS,
        'timestamp': current_time
    }

    results_path = f'{OUTPUT_DIR}/best_params_{current_time}.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"\nResults saved to: {results_path}")

    all_trials = []
    for trial in study.trials:
        trial_info = {
            'trial_number': trial.number + 1,
            'params': trial.params,
            'value': trial.value,
            'median_epoch': trial.user_attrs.get('median_epoch', None),
            'fold_best_epochs': trial.user_attrs.get('fold_best_epochs', []),
            'std_mcc': trial.user_attrs.get('std_mcc', None),
            'state': str(trial.state)
        }
        all_trials.append(trial_info)

    trials_path = f'{OUTPUT_DIR}/all_trials_{current_time}.json'
    with open(trials_path, 'w', encoding='utf-8') as f:
        json.dump(all_trials, f, indent=2, ensure_ascii=False)
    logger.info(f"All trials saved to: {trials_path}")

    if len(study.trials) >= 10:
        try:
            importance = optuna.importance.get_param_importances(study)
            logger.info("\nParameter Importance:")
            for param, imp in importance.items():
                logger.info(f"  {param}: {imp:.4f}")
        except Exception as e:
            logger.warning(f"Could not compute parameter importance: {e}")

    return study


def main():
    study = run_bayesian_optimization()

    logger.info("\n" + "=" * 70)
    logger.info("To use the best hyperparameters for final training:")
    logger.info("=" * 70)
    best = study.best_params
    best_median_epoch = study.best_trial.user_attrs.get('median_epoch', 200)
    logger.info(f"  batch_size = {best['batch_size']}")
    logger.info(f"  lr = {best['learning_rate']:.6e}")
    logger.info(f"  weight_decay = {best['weight_decay']:.6e}")
    logger.info(f"  epochs = {best_median_epoch}  (median of best epochs across CV folds)")


if __name__ == "__main__":
    main()
