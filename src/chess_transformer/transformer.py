import torch.nn as nn
import torch
import warnings

from .attention_layer import SelfAttentionLayer


class Transformer(nn.Module):
    def __init__(self, num_layers, num_heads, vocab_size, d_model, max_sequence_length):
        super().__init__()
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length

        # Weights
        self.self_attention_sublayers = nn.ModuleList(
            [SelfAttentionLayer(num_heads, d_model, max_sequence_length) 
             for _ in range(num_layers)])

        self.ffn_sublayers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, 2*d_model),
                nn.ReLU(),
                nn.Linear(2*d_model, d_model),
            )
            for _ in range(num_layers)
        ])

        self.final_linear = nn.Linear(d_model, vocab_size)

        # Positional encodings & embeddings
        pe_dim = d_model // (2 * num_heads)
        exponents = -torch.log(torch.tensor(10000)) * torch.arange(pe_dim) / pe_dim
        frequencies = torch.exp(exponents)
        angles = torch.arange(max_sequence_length)[:, None] * frequencies[None, :]

        sin_encodings = torch.sin(angles)
        cos_encodings = torch.cos(angles)
        self.register_buffer("pe_sin", sin_encodings)
        self.register_buffer("pe_cos", cos_encodings)

        self.embedding = nn.Embedding(vocab_size, d_model)

        # Normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model)
            for _ in range(2*num_layers)
        ])
        self.final_layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        T = x.shape[-1]
        x = self.embedding(x)

        for idx in range(self.num_layers):
            residual_connection = x
            x = self.self_attention_sublayers[idx](x, self.pe_sin[:T, ...], self.pe_cos[:T, ...])
            x = self.layer_norms[2*idx](x + residual_connection)
            residual_connection = x
            x = self.ffn_sublayers[idx](x)
            x = self.layer_norms[2*idx+1](x + residual_connection)

        return self.final_linear(x)

    def generate(self, x):
        if x.shape[-1] > self.max_sequence_length:
            warnings.warn(
                f"Context length ({x.shape[-1]}) exceeds max_sequence_length of transformer ({self.max_sequence_length});"
                f"Truncating input."
            )
            x = x[..., :self.max_sequence_length]

        return self.forward(x)