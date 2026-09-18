import torch
import torch.nn as nn

from attention import MultiheadAttention


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.attn = MultiheadAttention(d_model, num_heads)

        # TODO: two LayerNorm layers, one for each sublayer (pre norm style,
        # so each one normalizes *before* its sublayer runs)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        

        # TODO: the feedforward network. Two nn.Linear layers with your
        # chosen activation in between, d_model -> d_ff -> d_model
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x, mask=None, past_key_value=None):
        # TODO: attention sublayer, pre norm + residual:
        # x = x + self.attn(norm(x), norm(x), norm(x), mask)[0]
        # (remember MultiheadAttention returns a tuple)
        normed = self.norm1(x)
        attn_out, attn_weights, new_past_key_value = self.attn(
            normed, normed, normed, mask, past_key_value=past_key_value
        )

        # TODO: feedforward sublayer, same pre norm + residual pattern
        x = x + attn_out
        x = x + self.ff(self.norm2(x))

        return x, attn_weights, new_past_key_value