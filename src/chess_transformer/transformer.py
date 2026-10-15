import torch.nn as nn
import torch
import warnings

from .attention_layer import SelfAttentionLayer
import config

class Transformer(nn.Module):
    def __init__(self, num_layers, num_heads, input_vocab_size, output_vocab_size, d_model, max_sequence_length):
        super().__init__()
        self.num_layers = num_layers
        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size
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

        self.final_linear = nn.Linear(d_model, output_vocab_size)

        # Positional encodings & embeddings
        self.positional_encodings = nn.Parameter(torch.randn(config.BOARD_SEQUENCE_LENGTH, d_model) * 0.01)

        self.embedding = nn.Embedding(input_vocab_size, d_model)

        # Normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model)
            for _ in range(2*num_layers)
        ])

    def forward(self, x):
        x = self.embedding(x) + self.positional_encodings

        for idx in range(self.num_layers):
            residual_connection = x
            x = self.self_attention_sublayers[idx](x)
            x = self.layer_norms[2*idx](x + residual_connection)
            residual_connection = x
            x = self.ffn_sublayers[idx](x)
            x = self.layer_norms[2*idx+1](x + residual_connection)

        CLS_activation = self.final_linear(x)[:, -1, :]

        return CLS_activation