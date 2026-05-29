import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingMLPBranch(nn.Module):
    """
    ESM2 Embedding -> MLP branch
    Takes pre-computed ESM2 embeddings (1280 dim) and processes through MLP.
    """
    def __init__(self, embedding_dim=1280, hidden_dim=128, num_layers=3, center_idx=20):
        super().__init__()
        self.center_idx = center_idx

        self.input_proj = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.mlp_layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            self.mlp_layers.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 2),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(hidden_dim * 2, hidden_dim)
                )
            )
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

    def forward(self, embedding, return_tokens=False):
        # embedding: (B, L, 1280)
        x = self.input_proj(embedding)  # (B, L, 128)

        for i in range(len(self.mlp_layers)):
            residual = x
            x = self.mlp_layers[i](x)
            x = F.dropout(x, p=0.3, training=self.training)
            x = x + residual
            x = self.norms[i](x)

        tokens = self.output_proj(x)  # -> (B, L, 128)

        actual_center_idx = min(self.center_idx, tokens.size(1) - 1)
        center_feat = tokens[:, actual_center_idx, :]  # -> (B, 128)

        if return_tokens:
            return center_feat, tokens
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
        x_seq = x_conv.transpose(1, 2)

        if x_seq.size(1) != self.pos_embedding.size(1):
            pos_emb = self.pos_embedding.expand(x_seq.size(0), -1, -1)
            pos_emb = F.interpolate(pos_emb.transpose(1, 2), size=x_seq.size(1), mode='linear', align_corners=False)
            pos_emb = pos_emb.transpose(1, 2)
            x_seq = x_seq + pos_emb
        else:
            x_seq = x_seq + self.pos_embedding

        tokens = self.transformer(x_seq)

        center_feat = tokens[:, self.center_idx, :]
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
        gamma = self.to_gamma(cond).unsqueeze(1)
        beta = self.to_beta(cond).unsqueeze(1)

        return gamma * x + beta


class PLMEmbeddingModel(nn.Module):
    """
    Model using ESM2 embeddings + one-hot sequence features
    with FiLM cross-modal fusion
    """
    def __init__(self, embedding_dim=1280, hidden_dim=128, num_classes=2, center_idx=20):
        super().__init__()
        self.center_idx = center_idx

        self.embedding_branch = EmbeddingMLPBranch(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            center_idx=center_idx
        )

        self.seq_branch = OneHotCNNBranch(
            in_dim=21,
            out_dim=hidden_dim,
            center_idx=center_idx
        )

        self.film_seq = FiLM(hidden_dim)
        self.film_embedding = FiLM(hidden_dim)

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

    def forward(self, embeddings, seq_onehot=None, return_features=False):
        embedding_feat, embedding_tokens = self.embedding_branch(embeddings, return_tokens=True)

        if seq_onehot is None:
            seq_feat = torch.zeros_like(embedding_feat)
            seq_tokens = seq_feat.unsqueeze(1)
        else:
            seq_feat, seq_tokens = self.seq_branch(seq_onehot, return_tokens=True)

        seq_tokens = self.film_seq(seq_tokens, embedding_feat)
        
        embedding_tokens = self.film_embedding(embedding_tokens, seq_feat)

        center_idx_seq = min(self.center_idx, seq_tokens.size(1) - 1)
        seq_feat = seq_tokens[:, center_idx_seq, :]

        center_idx_emb = min(self.center_idx, embedding_tokens.size(1) - 1)
        embedding_feat = embedding_tokens[:, center_idx_emb, :]

        fused = torch.cat([embedding_feat, seq_feat], dim=-1)
        fused = self.fusion(fused)

        logits = self.classifier(fused)

        if return_features:
            return logits, fused, embedding_feat, seq_feat

        return logits