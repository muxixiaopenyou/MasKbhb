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


class GraphModel(nn.Module):
    def __init__(self, node_dim=1280, edge_dim=16, hidden_dim=128, num_classes=2):
        super().__init__()

        self.graph_branch = EmbeddingGraphBranch(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim
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

    def forward(self, graph_batch, return_features=False):
        graph_feat = self.graph_branch(graph_batch, return_tokens=False)

        logits = self.classifier(graph_feat)

        if return_features:
            return logits, graph_feat

        return logits
