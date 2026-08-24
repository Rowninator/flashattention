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

    def forward(self, q, k, v, mask=None):
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

        out, atten_weights = scaled_dot_product_attention(q, k, v, mask)

        out = out.transpose(1,2)

        out = out.contiguous().view(batch_size, -1, self.d_model)

        out = self.wo(out)

        return out, atten_weights


def copy_weights_into_torch_mha(my_mha, torch_mha, d_model):
    with torch.no_grad():
        torch_mha.in_proj_weight[:d_model] = my_mha.wq.weight
        torch_mha.in_proj_weight[d_model:2 * d_model] = my_mha.wk.weight
        torch_mha.in_proj_weight[2 * d_model:] = my_mha.wv.weight

        torch_mha.in_proj_bias[:d_model] = my_mha.wq.bias
        torch_mha.in_proj_bias[d_model:2 * d_model] = my_mha.wk.bias
        torch_mha.in_proj_bias[2 * d_model:] = my_mha.wv.bias

        torch_mha.out_proj.weight.copy_(my_mha.wo.weight)
        torch_mha.out_proj.bias.copy_(my_mha.wo.bias)


# Dimensions
batch_size = 2
seq_len = 5
d_model = 512
num_heads = 8

myMHA = MultiheadAttention(d_model, num_heads)

torch_mha = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=num_heads, 
                batch_first=True)

copy_weights_into_torch_mha(myMHA, torch_mha, d_model)


x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)


x_my = x.detach().clone().requires_grad_(True)
x_torch = x.detach().clone().requires_grad_(True)


# Forward pass
my_out, my_weights = myMHA(x_my, x_my, x_my)

torch_out, torch_weights = torch_mha(
    x_torch,
    x_torch,
    x_torch
)

forward_diff = (my_out - torch_out).abs().max()

# backward pass
my_out.sum().backward()
torch_out.sum().backward()


# compare grad
grad_diff = (x_my.grad - x_torch.grad).abs().max()


print("Maximum forward difference:", forward_diff.item())
print("Maximum gradient difference:", grad_diff.item())

print(
    "Forward match:",
    torch.allclose(my_out, torch_out, atol=1e-6, rtol=1e-5)
)

print(
    "Gradient match:",
    torch.allclose(x_my.grad, x_torch.grad, atol=1e-6, rtol=1e-5)
)