import torch
import torch.nn as nn
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model, dropout=0):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        slope = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * slope)  # even dimensions
        pe[:, 1::2] = torch.cos(position * slope)  # odd dimensions
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x is N, L, D
        # pe is 1, maxlen, D
        scaled_x = x * np.sqrt(self.d_model)
        encoded = scaled_x + self.pe[:, :x.size(1), :]
        #  TODO: Check if dropout here helps
        return encoded


class Transformer(nn.Module):
    """
    Model from "A detailed guide to Pytorch's nn.Transformer() module.", by
    Daniel Melchor: https://medium.com/@danielmelchor/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
    """

    # Constructor
    def __init__(self, n_features, hidden_dim=128, seq_len=288, narrow_attn_heads=0, num_layers=6, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        # LAYERS
        self.positional_encoding = PositionalEncoding(max_len=seq_len, d_model=hidden_dim, dropout=dropout)
        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=narrow_attn_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.proj = nn.Linear(n_features, hidden_dim)  # from n_features to encoder hidden dimensions
        self.linear = nn.Linear(hidden_dim, n_features)  # from decoder hidden dimensions to n_features

    def forward(self, X):
        device = next(self.parameters()).device
        # Src size must be (batch_size, src sequence length)
        # Tgt size must be (batch_size, tgt sequence length)
        source_sequence = X[:, :self.seq_len, :]
        shifted_target_sequence = X[:, self.seq_len-1:-1, :]

        # Positional encoding - Out size = (batch_size, sequence length, dim_model)
        source_sequence = self.positional_encoding(self.proj(source_sequence))
        target_sequence = self.positional_encoding(self.proj(shifted_target_sequence))
        target_mask = self.get_target_mask(size=target_sequence.shape[1]).to(device)

        # Transformer blocks - Out size = (sequence length, batch_size, num_tokens)
        transformer_out = self.transformer(source_sequence, target_sequence, tgt_mask=target_mask, src_key_padding_mask=None,
                                           tgt_key_padding_mask=None)
        out = self.linear(transformer_out)
        return out

    def get_target_mask(self, size) -> torch.tensor:
        # Generates a square matrix where each row allows one event more to be seen
        mask = torch.tril(torch.ones(size, size) == 1)  # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf'))  # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0))  # Convert ones to 0

        # EX for size=5:
        # [[0., -inf, -inf, -inf, -inf],
        #  [0.,   0., -inf, -inf, -inf],
        #  [0.,   0.,   0., -inf, -inf],
        #  [0.,   0.,   0.,   0., -inf],
        #  [0.,   0.,   0.,   0.,   0.]]

        return mask

    def create_pad_mask(self, matrix: torch.tensor, pad_token: int) -> torch.tensor:
        # If matrix = [1,2,3,0,0,0] where pad_token=0, the result mask is
        # [False, False, False, True, True, True]
        return (matrix == pad_token)