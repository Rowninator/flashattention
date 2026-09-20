import torch
import torch.nn as nn

from transformer_block import TransformerBlock
from positional_encoding import positional_encoding


class GPTStyleModel(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_len):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        pe = positional_encoding(max_len, d_model)
        self.register_buffer("pe", pe)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])
        self.output_projection = nn.Linear(d_model, vocab_size)

    def forward(self, token_ids, mask=None, past_key_values=None):
        seq_len = token_ids.shape[1]

        if past_key_values is None:
            # fresh start, one empty slot per block, nothing cached yet
            past_key_values = [None] * len(self.blocks)
            offset = 0
        else:
            # every layer's cache has grown to the same length so far,
            # so layer 0's key tensor tells us where the new token(s)
            # actually sit: right after whatever's already cached
            offset = past_key_values[0][0].shape[2]

        x = self.token_embedding(token_ids)
        x = x + self.pe[offset:offset + seq_len]

        new_past_key_values = []
        for block, past_kv in zip(self.blocks, past_key_values):
            x, new_kv = block(x, mask, past_key_value=past_kv)
            new_past_key_values.append(new_kv)

        logits = self.output_projection(x)

        return logits, new_past_key_values