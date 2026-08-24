import math

import torch


def scaled_dot_product_attention(q, k, v, mask=None):
    """
    Args:
        q: (batch_size, seq_len_q, d_k)
        k: (batch_size, seq_len_k, d_k)
        v: (batch_size, seq_len_k, d_v)
        mask: optional bool tensor broadcastable to (batch_size, seq_len_q, seq_len_k).
              True (or truthy) marks positions to KEEP; False marks positions to block.

    Returns:
        out: (batch_size, seq_len_q, d_v)
        attn_weights: (batch_size, seq_len_q, seq_len_k)
    """
    d_k = q.shape[-1]
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    attn_weights = torch.softmax(scores, dim=-1)
    out = attn_weights @ v
    return out, attn_weights
