import time
import torch

from model import GPTStyleModel
from attention import make_causal_mask


def generate_uncached(model, initial_ids, num_new_tokens, block_size, mask_fn):
    ids = initial_ids
    for _ in range(num_new_tokens):
        if ids.shape[1] > block_size:
            break
        mask = mask_fn(ids.shape[1])
        logits, _ = model(ids, mask)
        selected_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        ids = torch.cat([ids, selected_id], dim=1)
    return ids


def generate_cached(model, initial_ids, num_new_tokens, block_size):
    ids = initial_ids
    mask = make_causal_mask(initial_ids.shape[1])
    logits, past_key_values = model(ids, mask, past_key_values=None)
    selected_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
    ids = torch.cat([ids, selected_id], dim=1)
    for _ in range(num_new_tokens - 1):
        if ids.shape[1] > block_size:
            break
        logits, past_key_values = model(selected_id, mask=None, past_key_values=past_key_values)
        selected_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        ids = torch.cat([ids, selected_id], dim=1)
    return ids


def time_generation(fn, *args, **kwargs):
    start_time = time.perf_counter()
    fn(*args, **kwargs)
    end_time = time.perf_counter()
    return end_time - start_time


torch.manual_seed(0)
model = GPTStyleModel(vocab_size=50, d_model=128, num_heads=4, d_ff=512, num_layers=4, max_len=256)
model.eval()

block_size = 256

# --- warmup: pay the one-time process startup cost here, result thrown away ---
with torch.no_grad():
    warmup_ids = torch.randint(0, 50, (1, 4))
    generate_uncached(model, warmup_ids, 5, block_size, make_causal_mask)
    generate_cached(model, warmup_ids, 5, block_size)

# --- the real, timed comparison ---
for num_new_tokens in [10, 50, 100, 200]:
    initial_ids = torch.randint(0, 50, (1, 4))
    with torch.no_grad():
        uncached_time = time_generation(generate_uncached, model, initial_ids, num_new_tokens, block_size, make_causal_mask)
        cached_time = time_generation(generate_cached, model, initial_ids, num_new_tokens, block_size)
        print(f"Num new tokens: {num_new_tokens:>4} | Uncached: {uncached_time:.4f}s | Cached: {cached_time:.4f}s | Speedup: {uncached_time / cached_time:.2f}x")