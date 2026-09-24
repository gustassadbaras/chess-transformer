import torch.nn as nn
import torch

import math

class SelfAttentionLayer(nn.Module):
    def __init__(self, num_heads, d_model, max_sequence_length):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(
            "d_model must be divisible by num_heads: "
            f"got d_model={d_model}, num_heads={num_heads}"
        )
        self.d_KQ = d_model // num_heads
        self.d_V = d_model // num_heads
        self.num_heads = num_heads
        self.d_model = d_model

        self.sqrt_d_KQ = math.sqrt(self.d_KQ)

        self.W_K = nn.Linear(d_model, self.d_KQ * num_heads)
        self.W_Q = nn.Linear(d_model, self.d_KQ * num_heads)
        self.W_V = nn.Linear(d_model, self.d_V * num_heads)
        self.linear = nn.Linear(d_model, d_model)

        self.ReLU = nn.ReLU()
        attention_mask = torch.ones(max_sequence_length, max_sequence_length).triu(diagonal=1)
        self.register_buffer("attention_mask", attention_mask)

    def apply_rotations(self, x, pe_sin, pe_cos):
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        x_even_rotated = x_even * pe_cos - x_odd * pe_sin
        x_odd_rotated = x_even * pe_sin + x_odd * pe_cos

        x = torch.stack((x_even_rotated, x_odd_rotated), dim=-1).flatten(-2)

        return x

    def forward(self, x, pe_cos, pe_sin):
        B, T, _ = x.shape

        # Shapes should be (B, num_heads, T, d_KQ or d_V)
        Q = self.W_Q(x).view(B, T, self.num_heads, self.d_KQ).transpose(1, 2)
        K = self.W_K(x).view(B, T, self.num_heads, self.d_KQ).transpose(1, 2)
        V = self.W_V(x).view(B, T, self.num_heads, self.d_V).transpose(1, 2)

        Q = self.apply_rotations(Q, pe_sin, pe_cos)
        K = self.apply_rotations(K, pe_sin, pe_cos)

        scaled_attention_scores = torch.matmul(Q, K.transpose(-1,-2)) / self.sqrt_d_KQ
        masked_scaled_attention_scores = scaled_attention_scores.masked_fill(
            self.attention_mask[:T,:T] == 1,
            float("-inf")
        )

        attention_weights = torch.softmax(masked_scaled_attention_scores, dim=-1)
        weighted_values = torch.matmul(attention_weights, V)

        concat_values = weighted_values.transpose(1, 2).reshape(B, T, self.d_model)

        return self.linear(concat_values)