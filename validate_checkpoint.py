import torch
import torch.nn as nn
import tiktoken

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
val_ids = ids[split_idx:]

vocab_size = enc.n_vocab
block_size = 64
batch_size = 32
eval_iters = 10
mask = make_causal_mask(block_size)
loss_fn = nn.CrossEntropyLoss()

model = GPTStyleModel(vocab_size=vocab_size, d_model=128, num_heads=4, d_ff=512, num_layers=4, max_len=block_size)
model.load_state_dict(torch.load("model_checkpoint.pt", weights_only=True))

vloss = estimate_loss(model, val_ids, eval_iters, batch_size, block_size, mask, loss_fn)
print("Validation loss of the saved checkpoint:", vloss.item())
print("Compare against: best logged value ~4.836 (step 2200) vs. final step value ~5.089 (step 2999)")
