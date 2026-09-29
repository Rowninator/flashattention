import torch
from attention import scaled_dot_product_attention
from attention import make_causal_mask


seq_len = 4
d_model = 8

torch.manual_seed(0)

# TODO: build x1, a single sequence, shape (1, seq_len, d_model)
x1 = torch.randn(1, seq_len, d_model)

# TODO: build x2 as a copy of x1, but with position 3 (the last token)
# replaced by something completely different. Careful here: a plain
# assignment like x2 = x1 doesn't actually make an independent copy in
# torch, so look up how to clone a tensor before modifying one position.
x2 = x1.clone()

x2[:,3,:] = torch.randn(d_model)


mask = make_causal_mask(seq_len)

# TODO: run x1 through scaled_dot_product_attention as q=k=v=x1, passing
# the causal mask. Call the result out1.
out1, _ = scaled_dot_product_attention(x1,x1,x1, mask=mask)

# TODO: same thing for x2. Call the result out2.
out2, _ = scaled_dot_product_attention(x2,x2,x2, mask=mask)

# TODO: compare out1 and out2 at positions 0, 1, 2. What do you expect,
# and what function would you use to check "are these the same"?
print(torch.allclose(out1[:, :3, :], out2[:, :3, :]))


# TODO: compare out1 and out2 at position 3. What do you expect here,
# and why should it be different from the first three?
print(torch.allclose(out1[:, 3, :], out2[:, 3, :]))