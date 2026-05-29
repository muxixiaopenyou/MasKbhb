# Generate graphs with embeddings as node features
import os
import torch
import numpy as np
import logging
from Bio.PDB import PDBParser
from torch_geometric.data import Data
from utils import load_fasta

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()


def residue_to_aa1(res):
    """Convert three-letter amino acid code to one-letter."""
    AA3_TO_1 = {
        'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
        'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
        'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
        'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
        'MSE': 'M'
    }
    return AA3_TO_1.get(res.get_resname().upper(), 'X')


def locate_window_indices_from_window(raw_seq, nodes):
    """
    Locate indices of the window sequence within PDB residues

    raw_seq: window sequence
    nodes: sequence of PDB residue objects
    Returns a list of residue indices or None if not found
    """
    
    raw_seq = (raw_seq or '').upper()
    window_seq = raw_seq.replace('X', '')
    if not window_seq:
        return None

    pdb_seq = ''.join(residue_to_aa1(r) for r in nodes)
    start = pdb_seq.find(window_seq)
    if start == -1:
        return None

    return list(range(start, start + len(window_seq)))


def locate_anchor_idx_from_window(raw_seq, nodes):
    # Locate the anchor index of the window in PDB residues

    raw_seq = (raw_seq or '').upper()
    window_seq = raw_seq.replace('X', '')
    if not window_seq:
        return None

    pdb_seq = ''.join(residue_to_aa1(r) for r in nodes)
    start_positions = []
    start = pdb_seq.find(window_seq)
    while start != -1:
        start_positions.append(start)
        start = pdb_seq.find(window_seq, start + 1)

    if not start_positions:
        return None

    center_seq_idx = len(raw_seq) // 2
    anchor_offset = len(raw_seq[:center_seq_idx].replace('X', ''))
    if anchor_offset >= len(window_seq):
        return None

    return start_positions[0] + anchor_offset


def parse_protein_id(sample_id):
    # Parse protein ID from sample identifier.
    # pos_<Protein>_<number>

    sid = (sample_id or '').strip()
    if not sid:
        return sid

    parts = sid.split('_')
    if len(parts) >= 3 and parts[0].lower() == 'pos':
        return parts[1].strip()
    
    return sid


def sanitize_name(name):
    return (name or '').replace('/', '_').replace('\\', '_').replace('|', '_').strip()


def resolve_pdb_file(pdb_dirs, sid):
    protein_id = parse_protein_id(sid)
    candidates = []
    for name in [sid, protein_id]:
        safe_name = sanitize_name(name)
        if safe_name and safe_name not in candidates:
            candidates.append(safe_name)

    for pdb_dir in pdb_dirs:
        for candidate in candidates:
            pdb_file = os.path.join(pdb_dir, f"{candidate}.pdb")
            if os.path.exists(pdb_file):
                return pdb_file

    if not candidates or not pdb_dirs:
        return None
    return os.path.join(pdb_dirs[0], f"{candidates[0]}.pdb")


def build_embedding_graph(sid, raw_seq, label, pdb_path, embedding, radius=12.0):
    if not os.path.exists(pdb_path):
        return None

    parser = PDBParser(QUIET=True)
    struct = parser.get_structure(sid, pdb_path)
    nodes = []

    # only collect CA atoms
    for res in struct.get_residues():
        if 'CA' in res:
            nodes.append(res)

    if not nodes:
        return None

    coords = np.array([r['CA'].get_coord() for r in nodes])

    # locate residue indices corresponding to the window sequence
    target_indices = locate_window_indices_from_window(raw_seq, nodes)
    if target_indices is None:
        return None
    target_indices = np.array(target_indices, dtype=np.int64)

    # check embedding length matches the raw sequence length
    if embedding.shape[0] != len(raw_seq):
        logger.warning(f"Embedding length mismatch for {sid}: emb={embedding.shape[0]}, seq={len(raw_seq)}")
        return None

    # extract embeddings of valid residues in the window as node features
    # 'X' in raw_seq are placeholders and are filtered out
    valid_mask = [c != 'X' for c in raw_seq]
    valid_indices = [i for i, valid in enumerate(valid_mask) if valid]

    if len(valid_indices) != len(target_indices):
        logger.warning(f"Valid residue count mismatch for {sid}")
        return None

    node_features = []
    for i, seq_idx in enumerate(valid_indices):
        if seq_idx >= embedding.shape[0]:
            logger.warning(f"Sequence index out of range for {sid}")
            return None
        node_features.append(embedding[seq_idx])

    x = torch.stack(node_features, dim=0)  # (num_nodes, 1280)

    # build edges based on distance
    edge_index, edge_attr = [], []
    sub_coords = coords[target_indices]

    # RBF parameters
    num_kernels = 16
    centers = np.linspace(0, 8, num_kernels)
    step = centers[1] - centers[0]
    gamma = 1.0 / (step ** 2)

    for i in range(len(target_indices)):
        for j in range(i + 1, len(target_indices)):
            d = np.linalg.norm(sub_coords[i] - sub_coords[j])
            if d > radius:
                continue

            edge_feat = np.exp(-gamma * (d - centers) ** 2).tolist()

            edge_index.append([i, j])
            edge_attr.append(edge_feat)
            edge_index.append([j, i])
            edge_attr.append(edge_feat)

    anchor_idx = locate_anchor_idx_from_window(raw_seq, nodes)
    if anchor_idx is None:
        return None

    center_node_idx = None
    for local_idx, global_idx in enumerate(target_indices):
        if global_idx == anchor_idx:
            center_node_idx = local_idx
            break

    if center_node_idx is None:
        return None

    # if there are no edges, add a self-loop
    if not edge_index:
        edge_index = [[0, 0]]
        edge_attr = [[0.0] * num_kernels]

    return Data(
        x=x,  # node features: embedding (num_nodes, 1280)
        edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
        y=torch.tensor([label], dtype=torch.long),
        center_idx=torch.tensor([center_node_idx], dtype=torch.long)
    )


def main():
    data_dir = 'data'
    base_pdb_dir = 'new_pdb'

    for split in ['train', 'test']:
        fasta_path = os.path.join(data_dir, f'{split}.fasta')
        embedding_path = os.path.join(data_dir, f'{split}_embedding.pt')
        output_path = os.path.join(data_dir, f'{split}_embedding_graph_6.pt')

        logger.info(f"Processing {split}...")

        ids, seqs, labels = load_fasta(fasta_path)
        embedding_data = torch.load(embedding_path)

        pdb_dirs = [
            base_pdb_dir,
            os.path.join(base_pdb_dir, split)
        ]

        graph_dict = {}

        for i, (sid, seq, y) in enumerate(zip(ids, seqs, labels)):
            pdb_file = resolve_pdb_file(pdb_dirs, sid)
            if pdb_file is None:
                continue

            if i >= len(embedding_data['embeddings']):
                logger.warning(f"Missing embedding for {sid}")
                continue

            emb = embedding_data['embeddings'][i]

            g = build_embedding_graph(sid, seq, y, pdb_file, emb, radius=6.0)
            if g:
                graph_dict[sid] = g

        torch.save(graph_dict, output_path)
        logger.info(f"Saved {len(graph_dict)} graphs to {output_path}")


if __name__ == '__main__':
    main()
