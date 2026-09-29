import copy
import torch
import torch.nn as nn
import tiktoken
import matplotlib.pyplot as plt

from model import GPTStyleModel
from attention import make_causal_mask


def get_batch(ids, batch_size, block_size, device="cpu"):
    max_start = len(ids) - block_size - 1
    starts = torch.randint(0, max_start + 1, (batch_size,))
    x = torch.stack([ids[i:i + block_size] for i in starts])
    y = torch.stack([ids[i + 1:i + block_size + 1] for i in starts])
    return x.to(device), y.to(device)


def estimate_loss(model, ids, eval_iters, batch_size, block_size, mask, loss_fn):
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(eval_iters):
            vx, vy = get_batch(ids, batch_size, block_size)
            vlogits, _ = model(vx, mask)
            loss = loss_fn(vlogits.view(-1, vlogits.size(-1)), vy.view(-1))
            losses.append(loss.item())
    model.train()
    return torch.tensor(losses).mean()


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
eval_iters = 10

model = GPTStyleModel(vocab_size=vocab_size, d_model=128, num_heads=4, d_ff=512, num_layers=4, max_len=block_size)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
loss_fn = nn.CrossEntropyLoss()
mask = make_causal_mask(block_size)

history = []

best_val_loss = float("inf")
best_state = None
steps_without_improvement = 0
patience = 5
min_delta = 0.01

for step in range(n_steps):
    x, y = get_batch(train_ids, batch_size, block_size)

    logits, _ = model(x, mask)
    loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % log_every == 0 or step == n_steps - 1:
        vloss = estimate_loss(model, val_ids, eval_iters, batch_size, block_size, mask, loss_fn)

        history.append((step, loss.item(), vloss.item()))
        print(
            f"Step {step}: train loss = {loss.item():.4f} (ppl {torch.exp(loss).item():.2f}) "
            f"| val loss = {vloss.item():.4f} (ppl {torch.exp(vloss).item():.2f})"
        )

        if vloss.item() < best_val_loss - min_delta:
            best_val_loss = vloss.item()
            best_state = copy.deepcopy(model.state_dict())
            steps_without_improvement = 0
        else:
            steps_without_improvement += 1

        if steps_without_improvement >= patience:
            print(f"Early stopping at step {step} due to no improvement in validation loss.")
            break

if best_state is not None:
    model.load_state_dict(best_state)

torch.save(model.state_dict(), "model_checkpoint.pt")

with open("loss_log.csv", "w") as f:
    f.write("step,train_loss,val_loss\n")
    for s, tl, vl in history:
        f.write(f"{s},{tl},{vl}\n")

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

print("Saved loss_log.csv, loss_curve.png, and model_checkpoint.pt")
print("Best val loss achieved:", best_val_loss)