import torch.nn as nn
import torch
import warnings

from .attention_layer import SelfAttentionLayer


class Transformer(nn.Module):
    def __init__(self, num_layers, num_heads, vocab_size, d_model, max_sequence_length):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length

        # Weights
        self.self_attention_layers = nn.ModuleList(
            [SelfAttentionLayer(num_heads, d_model, max_sequence_length) 
             for _ in range(num_layers)])

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model*4),
            nn.ReLU(),
            nn.Linear(d_model*4, d_model)
        )

        self.linear = nn.Linear(d_model, vocab_size)

        # Positional encodings & embeddings
        half_d = d_model // 2
        exponents = -torch.log(torch.tensor(10000)) * torch.arange(half_d) / half_d
        frequencies = torch.exp(exponents)
        angles = torch.arange(max_sequence_length)[:, None] * frequencies[None, :]

        sinusoidal_encodings = torch.cat((torch.sin(angles), torch.cos(angles)), dim=1)
        self.register_buffer("sinusoidal_encodings", sinusoidal_encodings)

        self.Embedding = nn.Embedding(vocab_size, d_model)

        # Normalization
        self.LayerNorm1 = nn.LayerNorm(d_model)
        self.LayerNorm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        T = x.shape[-1]

        x = self.Embedding(x) + self.sinusoidal_encodings[None, :T, :]

        for layer in self.self_attention_layers:
            residual_connection = x
            x = layer(x)
            x = self.LayerNorm1(x + residual_connection)

        residual_connection = x

        x = self.ffn(x)
        x = self.LayerNorm2(x + residual_connection)

        x = self.linear(x)

        return x

    def generate(self, x):
        if x.shape[-1] > self.max_sequence_length:
            warnings.warn(
                f"Context length ({x.shape[-1]}) exceeds max_sequence_length of transformer ({self.max_sequence_length});"
                f"Truncating input."
            )
            x = x[..., :self.max_sequence_length]

        return self.forward(x)