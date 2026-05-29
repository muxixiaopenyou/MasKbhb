import os
import torch
import logging
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data

logger = logging.getLogger()

def load_fasta(fasta_path):
    ids, seqs, labels = [], [], []
    if not os.path.exists(fasta_path):
        logger.warning(f"File not found: {fasta_path}")
        return ids, seqs, labels

    with open(fasta_path, 'r', encoding='utf-8') as f:
        current_id = None
        current_seq = []
        for line in f:
            line = line.strip()
            if not line: continue
            if line.startswith('>'):
                if current_id:
                    ids.append(current_id)
                    seqs.append("".join(current_seq))
                    labels.append(1 if 'pos' in current_id.lower() else 0)
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)
        if current_id:
            ids.append(current_id)
            seqs.append("".join(current_seq))
            labels.append(1 if 'pos' in current_id.lower() else 0)
            
    return ids, seqs, labels

def load_embedding_pt(path):
    try:
        payload = torch.load(path, map_location='cpu', weights_only=False)
        ids = payload['ids']
        embs = payload['embeddings']
        seq_dict = {sid: embs[i].float() for i, sid in enumerate(ids)}
        return ids, seq_dict
    except Exception as e:
        logger.error(f"Error loading embedding {path}: {e}")
        return [], {}

def load_graph_pt(path):
    return torch.load(path, map_location='cpu', weights_only=False)

class MultiModalDataset(Dataset):
    def __init__(self, ids, seq_dict, graph_dict, labels):
        self.valid_ids = []
        self.labels_dict = {}
        
        for sid, label in zip(ids, labels):
            if sid in seq_dict and sid in graph_dict:
                self.valid_ids.append(sid)
                self.labels_dict[sid] = torch.tensor(label, dtype=torch.long)
        
        self.seq_dict = seq_dict
        self.graph_dict = graph_dict
        
        logger.info(f"Dataset initialized: {len(self.valid_ids)} samples available.")

    def __len__(self):
        return len(self.valid_ids)

    def __getitem__(self, idx):
        sid = self.valid_ids[idx]
        label = self.labels_dict[sid]
        
        seq_tensor = self.seq_dict[sid]
        graph_data = self.graph_dict[sid]
        
        if isinstance(graph_data, Data):
            graph = graph_data.clone()
            graph.y = label.view(1) 
        elif hasattr(graph_data, 'x'):
            graph = Data(
                x=graph_data.x,
                edge_index=graph_data.edge_index,
                edge_attr=getattr(graph_data, 'edge_attr', None),
                y=label.view(1)
            )
        else:
            raise ValueError(f"Invalid graph format for {sid}")

        return seq_tensor, graph, label

def multimodal_collate(batch):
    seqs = torch.stack([item[0] for item in batch], dim=0)
    graphs = Batch.from_data_list([item[1] for item in batch])
    labels = torch.stack([item[2] for item in batch], dim=0)
    return seqs, graphs, labels