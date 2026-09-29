import torch


def uptrain_gqa_from_mha(mha, gqa, num_heads, num_kv_heads, head_dim):
    group_size = num_heads // num_kv_heads

    with torch.no_grad():
        gqa.wq.weight.copy_(mha.wq.weight)
        gqa.wq.bias.copy_(mha.wq.bias)
        gqa.wo.weight.copy_(mha.wo.weight)
        gqa.wo.bias.copy_(mha.wo.bias)

        for name in ["wk", "wv"]:
            mha_layer = getattr(mha, name)
            gqa_layer = getattr(gqa, name)

            w = mha_layer.weight.view(num_heads, head_dim, -1)
            w = w.view(num_kv_heads, group_size, head_dim, -1)
            w = w.mean(dim=1)
            gqa_layer.weight.copy_(w.reshape(num_kv_heads * head_dim, -1))

            b = mha_layer.bias.view(num_heads, head_dim)
            b = b.view(num_kv_heads, group_size, head_dim)
            b = b.mean(dim=1)
            gqa_layer.bias.copy_(b.reshape(num_kv_heads * head_dim))