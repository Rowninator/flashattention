import torch
import torch.nn as nn
import tiktoken

from attention import MultiheadAttention, GroupedQueryAttention, make_causal_mask
from positional_encoding import positional_encoding
from uptrain import uptrain_gqa_from_mha


class LocalTransformerBlock(nn.Module):
    def __init__(self, attn, d_model, d_ff):
        super().__init__()
        self.attn = attn
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x, mask=None):
        normed = self.norm1(x)
        attn_out, _, _ = self.attn(normed, normed, normed, mask)
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x


class LocalGPTStyleModel(nn.Module):
    def __init__(self, blocks, vocab_size, d_model, max_len):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        pe = positional_encoding(max_len, d_model)
        self.register_buffer("pe", pe)
        self.blocks = nn.ModuleList(blocks)
        self.output_projection = nn.Linear(d_model, vocab_size)

    def forward(self, token_ids, mask=None):
        seq_len = token_ids.shape[1]
        x = self.token_embedding(token_ids)
        x = x + self.pe[:seq_len]
        for block in self.blocks:
            x = block(x, mask)
        return self.output_projection(x)


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
            logits = model(vx, mask)
            loss = loss_fn(logits.view(-1, logits.size(-1)), vy.view(-1))
            losses.append(loss.item())
    return torch.tensor(losses).mean()


vocab_size, d_model, num_heads, d_ff, num_layers, max_len = 50257, 128, 4, 512, 4, 64
num_kv_heads = 2
block_size = 64
batch_size = 32
eval_iters = 20  # more than train.py's, since this is a final reported number

enc = tiktoken.get_encoding("gpt2")
with open("data/tinyshakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()
ids = torch.tensor(enc.encode(text), dtype=torch.long)
split_idx = int(0.9 * len(ids))
val_ids = ids[split_idx:]

loss_fn = nn.CrossEntropyLoss()
mask = make_causal_mask(block_size)

# --- MHA model, real trained checkpoint ---
mha_blocks = [
    LocalTransformerBlock(MultiheadAttention(d_model, num_heads), d_model, d_ff)
    for _ in range(num_layers)
]
mha_model = LocalGPTStyleModel(mha_blocks, vocab_size, d_model, max_len)
state_dict = torch.load("model_checkpoint.pt", map_location="cpu", weights_only=True)
mha_model.load_state_dict(state_dict)

# --- GQA model: non-attention weights copied, attention uptrained ---
gqa_blocks = [
    LocalTransformerBlock(GroupedQueryAttention(d_model, num_heads, num_kv_heads), d_model, d_ff)
    for _ in range(num_layers)
]
gqa_model = LocalGPTStyleModel(gqa_blocks, vocab_size, d_model, max_len)

non_attn_keys = [k for k in state_dict.keys() if ".attn." not in k]
gqa_state_dict = gqa_model.state_dict()
for k in non_attn_keys:
    gqa_state_dict[k] = state_dict[k]
gqa_model.load_state_dict(gqa_state_dict)

head_dim = d_model // num_heads
for i in range(num_layers):
    uptrain_gqa_from_mha(
        mha_model.blocks[i].attn,
        gqa_model.blocks[i].attn,
        num_heads,
        num_kv_heads,
        head_dim,
    )

mha_loss = estimate_loss(mha_model, val_ids, eval_iters, batch_size, block_size, mask, loss_fn)
gqa_loss = estimate_loss(gqa_model, val_ids, eval_iters, batch_size, block_size, mask, loss_fn)

print(f"Original MHA:            val loss {mha_loss.item():.4f}  (ppl {torch.exp(mha_loss).item():.2f})")
print(f"Freshly uptrained GQA:    val loss {gqa_loss.item():.4f}  (ppl {torch.exp(gqa_loss).item():.2f})")
print(f"Perplexity increase from uptraining alone, no further training: "
      f"{(torch.exp(gqa_loss) - torch.exp(mha_loss)).item():.2f}")