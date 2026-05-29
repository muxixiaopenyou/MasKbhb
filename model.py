import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import to_dense_batch

class EmbeddingGraphBranch(nn.Module):
    def __init__(self, node_dim=1280, edge_dim=16, hidden_dim=128, num_layers=3, heads=4):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.node_proj = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.gat_layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            self.gat_layers.append(
                GATConv(
                    hidden_dim,
                    hidden_dim // heads,
                    heads=heads,
                    concat=True,
                    edge_dim=edge_dim,
                    dropout=0.4
                )
            )
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        self.dropout = 0.4

    def forward(self, graph_batch, return_tokens=False):
        x, edge_index, edge_attr = (
            graph_batch.x,
            graph_batch.edge_index,
            graph_batch.edge_attr
        )

        x = self.node_proj(x)

        for i in range(self.num_layers):
            residual = x
            x = self.gat_layers[i](x, edge_index, edge_attr)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + residual
            x = self.norms[i](x)

        x = self.output_proj(x)

        x_dense, mask = to_dense_batch(x, graph_batch.batch)

        center_idx = graph_batch.center_idx.view(-1)
        bsz = x_dense.size(0)
        batch_ids = torch.arange(bsz, device=x.device)

        center_feat = x_dense[batch_ids, center_idx]

        if return_tokens:
            return center_feat, x_dense, mask
        return center_feat


class OneHotCNNBranch(nn.Module):
    def __init__(self, in_dim=21, out_dim=128, center_idx=20, seq_len=41, num_transformer_layers=2):
        super().__init__()
        self.center_idx = center_idx
        
        self.conv3 = nn.Conv1d(in_dim, 64, kernel_size=3, padding='same')
        self.conv5 = nn.Conv1d(in_dim, 64, kernel_size=5, padding='same')
        self.conv7 = nn.Conv1d(in_dim, 64, kernel_size=7, padding='same')
        
        self.conv_proj = nn.Sequential(
            nn.Conv1d(64 * 3, out_dim, kernel_size=1),
            nn.BatchNorm1d(out_dim),
            nn.ReLU()
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, out_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=out_dim,
            nhead=4,
            dim_feedforward=out_dim * 2,
            dropout=0.6,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        self.proj = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim)
        )

    def forward(self, seq_onehot, return_tokens=False):
        x = seq_onehot.transpose(1, 2).float()  

        x3 = F.relu(self.conv3(x))
        x5 = F.relu(self.conv5(x))
        x7 = F.relu(self.conv7(x))
        x_cat = torch.cat([x3, x5, x7], dim=1)  

        x_conv = self.conv_proj(x_cat)          
        x_seq = x_conv.transpose(1, 2)          # -> (B, 41, 128)
        x_seq = x_seq + self.pos_embedding

        tokens = self.transformer(x_seq)        # -> (B, 41, 128)

        center_feat = tokens[:, self.center_idx, :]  # -> (B, 128)
        center_feat = self.proj(center_feat)

        if return_tokens:
            tokens = self.proj(tokens)
            return center_feat, tokens

        return center_feat

class FiLM(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.to_gamma = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

        self.to_beta = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, x, cond):
        """
        x:    (B, L, D)
        cond: (B, D)
        """
        gamma = self.to_gamma(cond).unsqueeze(1)
        beta = self.to_beta(cond).unsqueeze(1)

        return gamma * x + beta


class PLMGraphModel(nn.Module):
    def __init__(self, node_dim=1280, edge_dim=16, hidden_dim=128, num_classes=2):
        super().__init__()

        self.graph_branch = EmbeddingGraphBranch(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim
        )

        self.seq_branch = OneHotCNNBranch(
            in_dim=21,
            out_dim=hidden_dim,
            center_idx=20
        )

        # FiLM: graph -> seq
        self.film_seq = FiLM(hidden_dim)

        # FiLM: seq -> graph (optional symmetric)
        self.film_graph = FiLM(hidden_dim)

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, graph_batch, seq_onehot=None, return_features=False):

        # ---------------- graph ----------------
        graph_feat, graph_tokens, _ = self.graph_branch(graph_batch, return_tokens=True)

        # ---------------- seq ----------------
        if seq_onehot is None:
            seq_feat = torch.zeros_like(graph_feat)
            seq_tokens = seq_feat.unsqueeze(1)
        else:
            seq_feat, seq_tokens = self.seq_branch(seq_onehot, return_tokens=True)

        # ---------------- FiLM conditioning ----------------
        # graph → modulate seq tokens
        seq_tokens = self.film_seq(seq_tokens, graph_feat)

        # seq → modulate graph tokens
        graph_tokens = self.film_graph(graph_tokens, seq_feat)

        center_idx_seq = min(self.seq_branch.center_idx, seq_tokens.size(1) - 1)
        seq_feat = seq_tokens[:, center_idx_seq, :]

        center_idx_graph = graph_batch.center_idx.view(-1)
        bsz = graph_tokens.size(0)
        batch_ids = torch.arange(bsz, device=graph_tokens.device)
        graph_feat = graph_tokens[batch_ids, center_idx_graph]

        fused = torch.cat([graph_feat, seq_feat], dim=-1)
        fused = self.fusion(fused)

        logits = self.classifier(fused)

        if return_features:
            return logits, fused, graph_feat, seq_feat

        return logits