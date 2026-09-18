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

# print(scaled_dot_product_attention(3,3,1))

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
    
        q = q.view(batch_size, -1, self.num_heads, self.d_head)
        k = k.view(batch_size, -1, self.num_heads, self.d_head)
        v = v.view(batch_size, -1, self.num_heads, self.d_head)

        q = q.transpose(1,2)
        k = k.transpose(1,2)
        v = v.transpose(1,2)

        # TODO: if past_key_value is not None, it's a (past_k, past_v) tuple
        # from the previous call. Concatenate it onto this step's freshly
        # computed k and v. You already worked out which dimension this
        # needs to happen on, back when you traced through DynamicLayer,
        # just remember k and v have already been transposed by this point
        # in your function, so double check that dimension number still
        # points at seq_len rather than somewhere else now.
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        


        # TODO: build the new_past_key_value tuple to return, using k and v
        # *after* the concatenation above, so it carries the full history
        # forward, not just whatever came in on this one call
        new_past_key_value = (k, v)


        out, atten_weights = scaled_dot_product_attention(q, k, v, mask)

        out = out.transpose(1,2)

        out = out.contiguous().view(batch_size, -1, self.d_model)

        out = self.wo(out)

        return out, atten_weights, new_past_key_value




def make_causal_mask(seq_len):
    keep = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
    mask = torch.zeros(seq_len, seq_len)
    mask = mask.masked_fill(~keep, float('-inf'))
    return mask

