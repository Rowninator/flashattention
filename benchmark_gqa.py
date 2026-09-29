import torch
from attention import MultiheadAttention, GroupedQueryAttention, make_causal_mask


def cache_size_bytes(past_key_value):
    # TODO: past_key_value is a (k, v) tuple. Each tensor has an
    # .element_size() (bytes per number) and .nelement() (how many
    # numbers total). Return the total bytes both tensors together
    # actually occupy.
    k, v = past_key_value
    return k.element_size() * k.nelement() + v.element_size() * v.nelement()



d_model, num_heads, num_kv_heads = 512, 8, 2
seq_len = 200

mha = MultiheadAttention(d_model, num_heads)
gqa = GroupedQueryAttention(d_model, num_heads, num_kv_heads)

x = torch.randn(1, seq_len, d_model)
mask = make_causal_mask(seq_len)

with torch.no_grad():
    # TODO: run x through mha and through gqa (self attention, q=k=v=x),
    # capturing each one's returned cache
    mha_out, mha_attn_weights, mha_past_key_value = mha(x, x, x, mask=mask)
    gqa_out, gqa_attn_weights, gqa_past_key_value = gqa(x, x, x, mask=mask)

    # TODO: compute cache_size_bytes for each, print both, and print the
    # ratio between them
    mha_cache_size = cache_size_bytes(mha_past_key_value)
    gqa_cache_size = cache_size_bytes(gqa_past_key_value)
    print(f"Multihead Attention cache size: {mha_cache_size} bytes")
    print(f"Grouped Query Attention cache size: {gqa_cache_size} bytes")
    print(f"Ratio (GQA/MHA): {gqa_cache_size / mha_cache_size}")