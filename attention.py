import math
import torch
import torch.nn as nn


def scaled_dot_product_attention(q, k, v, mask=None):
    d_k = q.shape[-1]
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        scores += mask
    attn_weights = torch.softmax(scores, dim=-1)
    out = attn_weights @ v
    return out, attn_weights


class MultiheadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None, past_key_value=None):
        batch_size = q.shape[0]
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        q = q.view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        new_past_key_value = (k, v)
        out, attn_weights = scaled_dot_product_attention(q, k, v, mask)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        out = self.wo(out)
        return out, attn_weights, new_past_key_value


def make_causal_mask(seq_len):
    keep = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
    mask = torch.zeros(seq_len, seq_len)
    mask = mask.masked_fill(~keep, float('-inf'))
    return mask