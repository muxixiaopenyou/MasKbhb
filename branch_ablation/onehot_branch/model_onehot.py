import torch
import torch.nn as nn
import torch.nn.functional as F


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
        x_seq = x_conv.transpose(1, 2)  # -> (B, 41, 128)
        x_seq = x_seq + self.pos_embedding

        tokens = self.transformer(x_seq)  # -> (B, 41, 128)

        center_feat = tokens[:, self.center_idx, :]  # -> (B, 128)
        center_feat = self.proj(center_feat)

        if return_tokens:
            tokens = self.proj(tokens)
            return center_feat, tokens

        return center_feat


class OneHotModel(nn.Module):
    def __init__(self, in_dim=21, hidden_dim=128, num_classes=2, center_idx=20, seq_len=41):
        super().__init__()

        self.seq_branch = OneHotCNNBranch(
            in_dim=in_dim,
            out_dim=hidden_dim,
            center_idx=center_idx,
            seq_len=seq_len
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

    def forward(self, seq_onehot, return_features=False):
        seq_feat = self.seq_branch(seq_onehot, return_tokens=False)

        logits = self.classifier(seq_feat)

        if return_features:
            return logits, seq_feat

        return logits
