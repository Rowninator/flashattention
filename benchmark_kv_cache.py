
import torch
import time
from model import GPTStyleModel
from attention import make_causal_mask

def generate_uncached(model, initial_ids, num_new_tokens, block_size, mask_fn):
    ids = initial_ids
    for _ in range(num_new_tokens):
        # TODO: run the full sequence so far through the model, no cache.
        # Get the mask for whatever the current sequence length is, take
        # the logits at the final position, and pick the next token id
        # (argmax is fine, don't worry about sampling strategies here).
        # Append it onto ids and continue.
        mask = mask_fn(ids.shape[1])
        if ids.shape[1] > block_size:
                    break
        logits = model(ids, mask)
        selected_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        
        ids = torch.cat([ids, selected_id], dim=1)



    return ids


def generate_cached(model, initial_ids, num_new_tokens, block_size):
    # TODO: prefill using the full initial_ids at once (needs a real causal
    # mask, since the prompt has more than one token), capturing the
    # resulting cache. Then loop num_new_tokens times, each time running
    # ONLY the single most recently generated token through the model,
    # mask=None, passing the cache forward and getting an updated one back.
    # Same next-token selection as above, append it, continue.
    ids = initial_ids
    

    # Step 1: process the whole starting prompt at once, get the cache back
    mask = make_causal_mask(initial_ids.shape[1])
    logits, past_key_values = model(ids, mask, past_key_values=None)

    # pick the first new token, same as generate_uncached did
    selected_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
    ids = torch.cat([ids, selected_id], dim=1)

    # Step 2: one token at a time from here, using the cache
    for _ in range(num_new_tokens - 1):
        if ids.shape[1] > block_size:
            break

        # TODO: run the model on ONLY selected_id (not all of ids),
        # mask=None, passing past_key_values in, and capture both
        # things it returns
        logits, past_key_values = model(selected_id, mask=None, past_key_values=past_key_values)


        # TODO: pick the next token the same way as above, and
        # append it to ids
        selected_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        ids = torch.cat([ids, selected_id], dim=1)
    
    return ids




# import generate_uncached and generate_cached from wherever you saved them


def time_generation(fn, *args, **kwargs):
    # TODO: record a start time, call fn(*args, **kwargs), record an end
    # time, and return the elapsed time. Look into time.perf_counter()
    # specifically, rather than time.time(), for measuring short
    # durations, one is meant for this, the other for wall clock dates.
    start_time = time.perf_counter()
    result = fn(*args, **kwargs)
    end_time = time.perf_counter()
    return end_time - start_time


torch.manual_seed(0)
model = GPTStyleModel(vocab_size=50, d_model=128, num_heads=4, d_ff=512, num_layers=4, max_len=256)
model.eval()

block_size = 256 
# TODO: pick your num_new_tokens values here, based on what you just
# reasoned through above

# --- warmup: pay the one-time process startup cost here, result thrown away ---
with torch.no_grad():
     warmup_ids = torch.randint(0, 50, (1, 4))
     generate_uncached(model, warmup_ids, num_new_tokens=5, block_size=256, mask_fn=make_causal_mask)
     generate_cached(model, warmup_ids, num_new_tokens=5, block_size=256)


# --- the real, timed comparison ---
for num_new_tokens in [10, 50, 100, 200]:
    initial_ids = torch.randint(0, 50, (1, 4))
    

    with torch.no_grad():
        # TODO: time generate_uncached and generate_cached, same
        # initial_ids, same num_new_tokens, same block_size for both,
        # so the comparison is actually fair
        uncached_time = time_generation(generate_uncached, model, initial_ids, num_new_tokens, block_size, make_causal_mask)
        cached_time = time_generation(generate_cached, model, initial_ids, num_new_tokens, block_size)

        # TODO: print both times, and the speedup ratio between them
        print(f"Num new tokens: {num_new_tokens}")
        print(f"Uncached time: {uncached_time}")
        print(f"Cached time: {cached_time}")
        print(f"Speedup ratio: {uncached_time / cached_time}")