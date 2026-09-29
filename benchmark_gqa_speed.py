import time
import torch
from attention import MultiheadAttention, GroupedQueryAttention, make_causal_mask


def generate_with_cache(layer, initial_x, num_new_steps, mask_fn):
    x = initial_x
    mask = mask_fn(x.shape[1])
    # TODO: prefill, same pattern as generate_cached before, capture the cache
    with torch.no_grad():
        out, attn_weights, past_key_value = layer(x, x, x, mask=mask)   

    for _ in range(num_new_steps):
        # TODO: build one new random tensor standing in for a new token,
        # same batch size and d_model as initial_x, but seq_len of 1
        new_token = torch.randn(x.shape[0], 1, x.shape[2])


        # TODO: run it through the layer, mask=None, passing the cache
        # forward, same pattern as before
        with torch.no_grad():
            out, attn_weights, past_key_value = layer(new_token, new_token, new_token, mask=None, past_key_value=past_key_value)


            
        


    return None  # nothing meaningful to return here, only the timing matters


def best_of(fn, *args, repeats=8, **kwargs):
    # TODO: call fn(*args, **kwargs) `repeats` separate times, timing each
    # one individually with time.perf_counter() (not time.time(), same
    # reasoning as before). Collect all the individual times, then return
    # the SMALLEST one, not the average, given what you now know about why
    # minimum is the right choice for timing specifically.
    times = []
    for _ in range(repeats):
        start_time = time.perf_counter()
        fn(*args, **kwargs)
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    return min(times)


d_model, num_heads, num_kv_heads = 512, 8, 2
mha = MultiheadAttention(d_model, num_heads)
gqa = GroupedQueryAttention(d_model, num_heads, num_kv_heads)

# TODO: same warmup step as the KV cache benchmark, one throwaway call to
# each layer before any real timing starts
with torch.no_grad():
    x = torch.randn(1, 10, d_model)
    mask = make_causal_mask(x.shape[1])
    mha(x, x, x, mask=mask)
    gqa(x, x, x, mask=mask)

# TODO: loop over a few different num_new_steps values, same as before,
# timing generate_with_cache for mha and for gqa, same starting input,
# same steps, printing both times and the speedup ratio
num_new_steps_values = [10, 20, 50]
for num_new_steps in num_new_steps_values:
    initial_x = torch.randn(1, 10, d_model)

    mha_time = best_of(generate_with_cache, mha, initial_x, num_new_steps, make_causal_mask)
    gqa_time = best_of(generate_with_cache, gqa, initial_x, num_new_steps, make_causal_mask)

    print(f"Num new steps: {num_new_steps}")
    print(f"Multihead Attention time: {mha_time:.6f} seconds")
    print(f"Grouped Query Attention time: {gqa_time:.6f} seconds")
    print(f"Speedup (MHA/GQA): {mha_time / gqa_time:.2f}x")