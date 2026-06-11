import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
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
from model_final import *
from utils import load_fasta
import matplotlib.pyplot as plt

os.makedirs('model', exist_ok=True)

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f'model/training_{current_time}.log'
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


def save_loss_history_csv(rows, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'lr'])
        writer.writerows(rows)


def save_loss_curve_png(rows, save_path, title='Loss Curve'):
    if plt is None:
        return False
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    epochs = [r[0] for r in rows]
    train_loss = [r[1] for r in rows]

    plt.figure(figsize=(8, 4.5))
    plt.plot(epochs, train_loss, label='train_loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    return True


def main():
    batch_size = 32
    lr = 3.500309e-05
    weight_decay = 2.603109e-03
    n_epochs = 18
    seed = 0
    data_dir = 'data'

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("="*30)
    logger.info("PLM Graph Model Training")
    logger.info("="*30)
    logger.info("Hyperparameters:")
    logger.info(f"Batch Size: {batch_size}")
    logger.info(f"Learning Rate: {lr}")
    logger.info(f"Weight Decay: {weight_decay}")
    logger.info(f"Epochs: {n_epochs}")
    logger.info(f"Node Feature Dimension: 1280")
    logger.info(f"Sequence Feature Dimension: 21 (20 AA + 1 X)")
    logger.info(f"Hidden Dimension: 128")
    logger.info("="*30)

    train_graph_path = os.path.join(data_dir, 'train_embedding_graph.pt')
    test_graph_path = os.path.join(data_dir, 'test_embedding_graph.pt')

    logger.info(f"Loading training data from {train_graph_path}...")
    train_graph_dict = torch.load(train_graph_path)
    test_graph_dict = torch.load(test_graph_path)

    train_fasta = os.path.join(data_dir, 'train.fasta')
    test_fasta = os.path.join(data_dir, 'test.fasta')
    train_ids, train_seqs, _ = load_fasta(train_fasta)
    test_ids, test_seqs, _ = load_fasta(test_fasta)
    train_seq_dict = {sid: seq for sid, seq in zip(train_ids, train_seqs)}
    test_seq_dict = {sid: seq for sid, seq in zip(test_ids, test_seqs)}

    train_dataset = GraphDataset(train_graph_dict, train_seq_dict)
    test_dataset = GraphDataset(test_graph_dict, test_seq_dict)

    if len(train_dataset) == 0:
        raise ValueError("No training samples after matching graph data with FASTA sequences.")
    if len(test_dataset) == 0:
        raise ValueError("No test samples after matching graph data with FASTA sequences.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=graph_collate
    )

    criterion = nn.CrossEntropyLoss()

    model = PLMGraphModel(node_dim=1280, edge_dim=16, hidden_dim=128, num_classes=2).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    loss_rows = []

    for epoch in range(n_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        current_lr = float(optimizer.param_groups[0]['lr'])
        logger.info(
            f"Epoch {epoch+1}/{n_epochs} - LR: {current_lr:.6g} - "
            f"Train Loss: {train_loss:.3f}"
        )

        loss_rows.append([
            epoch + 1,
            float(train_loss),
            float(current_lr),
        ])

    save_path = "./model/final_model.pth"
    torch.save(model.state_dict(), save_path)
    logger.info(f"Model saved to {save_path}")

    history_png = "./model/training_loss.png"
    save_loss_curve_png(loss_rows, history_png, title="Training Loss Curve")
    logger.info(f"Saved loss curve: {history_png}")

    logger.info("="*30)
    logger.info("Evaluating on test set...")

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=graph_collate
    )

    test_metrics, test_probs, test_labels = evaluate_model(model, test_loader, criterion, device)
    test_preds = (test_probs >= 0.5).astype(int)

    save_predictions(test_dataset.ids, test_labels, test_probs, test_preds,
                     'models/test_predictions.csv')

    logger.info("Test Metrics:")
    for k, v in test_metrics.items():
        if k != 'Val Loss':
            logger.info(f"{k:10}: {v:.3f}")


if __name__ == "__main__":
    main()
