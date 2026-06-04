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
import csv
import random
import logging
from datetime import datetime
from torch_geometric.data import Batch
from model_transformer_FiLM import *
from utils import load_fasta
import matplotlib.pyplot as plt

DISTANCES = [6]

AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_ORDER)}
X_IDX = len(AA_ORDER)
logger = logging.getLogger()


def extract_protein_name(fasta_id):
    parts = fasta_id.split('_')
    if len(parts) >= 2:
        return parts[1]
    return fasta_id


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
    return metrics, np.array(all_probs), np.array(all_labels)


def save_predictions(ids, labels, probs, preds, save_path):
    probs = np.array(probs).flatten().tolist()
    preds = np.array(preds).flatten().tolist()
    labels = np.array(labels).flatten().tolist()

    with open(save_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'label', 'prob', 'pred'])
        for i, sid in enumerate(ids):
            writer.writerow([
                str(sid),
                int(labels[i]),
                float(probs[i]),
                int(preds[i])
            ])


def run_distance_experiment(distance, batch_size, lr, weight_decay, n_epochs, k_folds,
                            patience, early_stop_metric, early_stop_min_delta, seed,
                            lr_scheduler_factor, lr_scheduler_patience, min_lr, device):
    DATA_TAG = f"{distance}"
    data_dir = 'data'

    models_dir = f'models_{DATA_TAG}'
    os.makedirs(models_dir, exist_ok=True)

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f'{models_dir}/training_{current_time}_{DATA_TAG}.log'

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    file_handler = logging.FileHandler(log_filename)
    stream_handler = logging.StreamHandler()
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[file_handler, stream_handler],
        force=True
    )

    logger.info("=" * 60)
    logger.info(f"DISTANCE: {distance}")
    logger.info("=" * 60)
    logger.info("Hyperparameters:")
    logger.info(f"  Batch Size: {batch_size}")
    logger.info(f"  Learning Rate: {lr}")
    logger.info(f"  Weight Decay: {weight_decay}")
    logger.info(f"  Epochs: {n_epochs}")
    logger.info(f"  K-Folds: {k_folds}")
    logger.info(f"  Patience: {patience}")
    logger.info(f"  Early Stop Metric: {early_stop_metric}")
    logger.info(f"  Node Feature Dimension: 1280")
    logger.info(f"  Sequence Feature Dimension: 21 (20 AA + 1 X)")
    logger.info(f"  Hidden Dimension: 128")
    logger.info(f"  LR Scheduler: ReduceLROnPlateau(factor={lr_scheduler_factor}, "
                f"patience={lr_scheduler_patience}, min_lr={min_lr})")
    logger.info(f"  Cross-Validation: GroupKFold (group = protein name)")
    logger.info("=" * 60)

    train_graph_path = os.path.join(data_dir, f'train_embedding_graph_{DATA_TAG}.pt')
    test_graph_path = os.path.join(data_dir, f'test_embedding_graph_{DATA_TAG}.pt')

    logger.info(f"Loading training data from {train_graph_path}...")
    train_graph_dict = torch.load(train_graph_path)
    test_graph_dict = torch.load(test_graph_path)

    train_fasta = os.path.join(data_dir, 'train.fasta')
    test_fasta = os.path.join(data_dir, 'test.fasta')
    train_ids, train_seqs, _ = load_fasta(train_fasta)
    test_ids, test_seqs, _ = load_fasta(test_fasta)
    train_seq_dict = {sid: seq for sid, seq in zip(train_ids, train_seqs)}
    test_seq_dict = {sid: seq for sid, seq in zip(test_ids, test_seqs)}

    full_train_dataset = GraphDataset(train_graph_dict, train_seq_dict)
    test_dataset = GraphDataset(test_graph_dict, test_seq_dict)

    if len(full_train_dataset) == 0:
        raise ValueError("No training samples after matching graph data with FASTA sequences.")
    if len(test_dataset) == 0:
        raise ValueError("No test samples after matching graph data with FASTA sequences.")

    groups = np.array([extract_protein_name(sid) for sid in full_train_dataset.ids])
    unique_proteins = np.unique(groups)
    logger.info(f"Number of unique proteins (groups): {len(unique_proteins)}")
    logger.info(f"Proteins: {', '.join(unique_proteins)}")

    y_for_cv = np.array([graph.y.item() for graph in full_train_dataset.graphs])
    criterion = nn.CrossEntropyLoss()

    group_kfold = GroupKFold(n_splits=k_folds)
    fold_metrics = []
    fold_paths = []

    x_dummy = np.zeros(len(y_for_cv), dtype=np.int64)
    for fold, (train_idx, val_idx) in enumerate(group_kfold.split(x_dummy, y_for_cv, groups)):
        train_proteins = set(groups[train_idx])
        val_proteins = set(groups[val_idx])
        overlap = train_proteins & val_proteins
        if overlap:
            logger.warning(f"Fold {fold+1}: Protein overlap detected! {overlap}")
        logger.info(f"===== Fold {fold + 1}/{k_folds} (train proteins: {len(train_proteins)}, "
                    f"val proteins: {len(val_proteins)}) =====")

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

        model = PLMGraphModel(node_dim=1280, edge_dim=16, hidden_dim=128, num_classes=2).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        scheduler_mode = 'max' if early_stop_metric != 'Val Loss' else 'min'
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=scheduler_mode,
            factor=lr_scheduler_factor,
            patience=lr_scheduler_patience,
            threshold=early_stop_min_delta,
            threshold_mode='abs',
            min_lr=min_lr,
            verbose=False
        )

        best_fold_metrics = None
        best_val_score = -float('inf')
        patience_counter = 0
        save_path = f"{models_dir}/fold_{fold+1}.pth"

        loss_rows = []  # epoch, train_loss, val_loss, val_score, lr

        for epoch in range(n_epochs):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_res, _, _ = evaluate_model(model, val_loader, criterion, device)

            current_lr = float(optimizer.param_groups[0]['lr'])
            logger.info(
                f"Epoch {epoch+1}/{n_epochs} - LR: {current_lr:.6g} - "
                f"Train Loss: {train_loss:.4f} - Val Loss: {val_res['Val Loss']:.4f}"
            )

            current_val_score = float(val_res[early_stop_metric])

            loss_rows.append([
                epoch + 1,
                float(train_loss),
                float(val_res['Val Loss']),
                float(current_val_score),
                float(current_lr),
            ])

            scheduler.step(current_val_score)

            if current_val_score > best_val_score + early_stop_min_delta:
                best_val_score = current_val_score
                best_fold_metrics = val_res
                torch.save(model.state_dict(), save_path)
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                logger.info(
                    f"Early stopping at epoch {epoch+1} (best {early_stop_metric}={best_val_score:.4f})"
                )
                break

        fold_metrics.append(best_fold_metrics)
        fold_paths.append(save_path)

        logger.info(f"Fold {fold+1} Best Metrics:")
        for k, v in best_fold_metrics.items():
            if k != 'Val Loss':
                logger.info(f"  {k:10}: {v:.4f}")

    logger.info("=" * 30)
    logger.info("Cross-Validation Results:")
    metric_names = ['MCC', 'SN', 'SP', 'ACC', 'BACC', 'PRECISION', 'F1', 'AUC', 'AUCPR']
    cv_summary = {}
    for name in metric_names:
        values = [m[name] for m in fold_metrics]
        mean_val, std_val = np.mean(values), np.std(values)
        cv_summary[name] = (mean_val, std_val)
        logger.info(f"  {name:10}: {mean_val:.4f} ± {std_val:.4f}")

    logger.info("=" * 30)
    logger.info("Evaluating ensemble on test set...")

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=graph_collate
    )

    all_fold_probs = []
    y_test_true = None

    for path in fold_paths:
        eval_model = PLMGraphModel(node_dim=1280, edge_dim=16, hidden_dim=128, num_classes=2).to(device)
        eval_model.load_state_dict(torch.load(path, map_location=device))

        _, probs, y_true = evaluate_model(eval_model, test_loader, criterion, device)
        all_fold_probs.append(probs)
        y_test_true = y_true

    ensemble_probs = np.mean(all_fold_probs, axis=0)
    ensemble_preds = (ensemble_probs >= 0.5).astype(int)

    final_test_metrics = calculate_metrics(y_test_true, ensemble_preds, ensemble_probs)

    test_ids = test_dataset.ids
    save_predictions(test_ids, y_test_true, ensemble_probs, ensemble_preds,
                     f'{models_dir}/test_predictions_{DATA_TAG}.csv')

    logger.info("Final Test Metrics (Ensemble):")
    for k, v in final_test_metrics.items():
        logger.info(f"  {k:10}: {v:.4f}")

    return {
        'distance': distance,
        'cv_summary': cv_summary,
        'test_metrics': final_test_metrics,
        'fold_metrics': fold_metrics,
    }


def main():
    batch_size = 128
    lr = 1e-4
    weight_decay = 1e-3
    n_epochs = 200
    k_folds = 5
    patience = 20
    early_stop_metric = 'MCC'
    early_stop_min_delta = 1e-4
    seed = 0

    lr_scheduler_factor = 0.5
    lr_scheduler_patience = 3
    min_lr = 1e-6

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_results = []

    for dist in DISTANCES:
        result = run_distance_experiment(
            distance=dist,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            n_epochs=n_epochs,
            k_folds=k_folds,
            patience=patience,
            early_stop_metric=early_stop_metric,
            early_stop_min_delta=early_stop_min_delta,
            seed=seed,
            lr_scheduler_factor=lr_scheduler_factor,
            lr_scheduler_patience=lr_scheduler_patience,
            min_lr=min_lr,
            device=device,
        )
        all_results.append(result)

    os.makedirs('log', exist_ok=True)
    summary_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = f'log/summary_{summary_time}.log'
    with open(summary_path, 'w', encoding='utf-8') as sf:

        def slog(msg):
            sf.write(msg + '\n')
            print(msg)

        slog("=" * 70)
        slog("FINAL COMPARISON: DISTANCE vs PERFORMANCE")
        slog("=" * 70)

        metric_names = ['MCC', 'SN', 'SP', 'ACC', 'BACC', 'PRECISION', 'F1', 'AUC', 'AUCPR']

        slog("\n--- Cross-Validation Results (mean +/- std across folds) ---")
        header = f"{'Dist':<6}" + "".join(f"{m:>14}" for m in metric_names)
        slog(header)
        slog("-" * len(header))
        for r in all_results:
            dist = r['distance']
            row = f"{dist:<6}"
            for m in metric_names:
                mean_v, std_v = r['cv_summary'][m]
                row += f"  {mean_v:.4f} ± {std_v:.3f}"
            slog(row)

        slog("\n--- Test Set Results (Ensemble) ---")
        header = f"{'Dist':<6}" + "".join(f"{m:>10}" for m in metric_names)
        slog(header)
        slog("-" * len(header))
        for r in all_results:
            dist = r['distance']
            row = f"{dist:<6}"
            for m in metric_names:
                row += f"  {r['test_metrics'][m]:.4f} "
            slog(row)

if __name__ == "__main__":
    main()
