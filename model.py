import torch
import torch.nn as nn

from transformer_block import TransformerBlock
from positional_encoding import positional_encoding


class GPTStyleModel(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_len):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # TODO: build the positional encoding table once, up to max_len,
        # and register it as a buffer so it moves with the model to a
        # device but never gets treated as a trainable parameter
        pe = positional_encoding(max_len, d_model)
        self.register_buffer("pe", pe)

        # TODO: a stack of num_layers TransformerBlocks. A plain Python
        # list won't register the submodules properly, look into
        # nn.ModuleList for this
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])

        # TODO: the output projection, d_model -> vocab_size
        self.output_projection = nn.Linear(d_model, vocab_size)

    def forward(self, token_ids, mask=None):
        seq_len = token_ids.shape[1]

        # TODO: token embedding + positional encoding (sliced to seq_len),
        # combined the way we just talked through
        x = self.token_embedding(token_ids)
        x = x + self.pe[:seq_len]

        # TODO: run x through every block in the stack, in order
        for block in self.blocks:
            x = block(x, mask)

        # TODO: project to vocab_size and return the logits
        logits = self.output_projection(x)


        return logits