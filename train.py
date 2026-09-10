import torch
import torch.nn as nn
import tiktoken

from model import GPTStyleModel
from attention import scaled_dot_product_attention, MultiheadAttention
from attention import make_causal_mask

def get_batch(ids, batch_size, block_size, device="cpu"):
    # ids: a 1D tensor of token ids for the dataset (or a split of it)

    # TODO: sample `batch_size` random starting indices. Think carefully
    # about the valid range here: each starting index needs enough room
    # left after it in `ids` for both the input window AND the shifted
    # target window to fit without running past the end.
    max_start = len(ids) - block_size - 1

    starts = torch.randint(0, max_start +1, (batch_size,))


    # TODO: build the input and target tensors by stacking one window per
    # sampled starting index, using the shift relationship from before
    # (target = input, shifted forward by one)
    x = torch.stack([ids[i:i + block_size] for i in starts])
    y = torch.stack([ids[i+1:i + block_size + 1] for i in starts])

    # TODO: move both tensors to `device` and return them
    x = x.to(device)
    y = y.to(device)

    return x, y



enc = tiktoken.get_encoding("gpt2")

with open("data/tinyshakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()

ids = torch.tensor(enc.encode(text), dtype=torch.long)

split_idx = int(0.9 * len(ids))
train_ids = ids[:split_idx]
val_ids = ids[split_idx:]

vocab_size = enc.n_vocab
block_size = 64
batch_size = 32

model = GPTStyleModel(vocab_size=vocab_size, d_model=128, num_heads=4, d_ff=512, num_layers=4, max_len=block_size)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
loss_fn = nn.CrossEntropyLoss()
mask = make_causal_mask(block_size)

for step in range(3000):
    x, y = get_batch(train_ids, batch_size, block_size)

    logits = model(x, mask)

    loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


    if step % 200 == 0:
        print(f"Step {step}: loss = {loss.item():.4f}")
        