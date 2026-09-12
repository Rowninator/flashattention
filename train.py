import torch
import torch.nn as nn
import tiktoken

from model import GPTStyleModel
from attention import scaled_dot_product_attention, MultiheadAttention
from attention import make_causal_mask

import matplotlib.pyplot as plt

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
n_steps = 3000
log_every = 200

model = GPTStyleModel(vocab_size=vocab_size, d_model=128, num_heads=4, d_ff=512, num_layers=4, max_len=block_size)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
loss_fn = nn.CrossEntropyLoss()
mask = make_causal_mask(block_size)

history = []

for step in range(3000):
    x, y = get_batch(train_ids, batch_size, block_size)

    logits = model(x, mask)

    loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


    if step % log_every == 0 or step == n_steps -1:
        model.eval()
        with torch.no_grad():
            vx, vy = get_batch(val_ids, batch_size, block_size)
            vlogits = model(vx, mask)
            vloss = loss_fn(vlogits.view(-1, vlogits.size(-1)), vy.view(-1))
        model.train()

        history.append((step, loss.item(), vloss.item()))

with open("loss_log.csv", "w") as f:
    f.write("step,train_loss,val_loss\n")
    for step, train_loss, val_loss in history:
        f.write(f"{step},{train_loss},{val_loss}\n")



steps = [h[0] for h in history]
train_losses = [h[1] for h in history]
val_losses = [h[2] for h in history]
 
plt.figure(figsize=(8, 5))
plt.plot(steps, train_losses, label="train loss")
plt.plot(steps, val_losses, label="val loss")
plt.xlabel("step")
plt.ylabel("cross entropy loss")
plt.title("Training loss, tiny Shakespeare")
plt.legend()
plt.tight_layout()
plt.savefig("loss_curve.png")
 
print("Saved loss_log.csv and loss_curve.png")