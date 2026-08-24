import numpy as np
import matplotlib.pyplot as plt

def positional_encoding(seq_len, d_model):
    PE = np.zeros((seq_len, d_model))
    pos = np.arange(seq_len)[:, None]          # (seq_len, 1)
    i = np.arange(0, d_model, 2)               # pair index: 0,2,4,...
    denom = 10000 ** (i / d_model)
    PE[:, 0::2] = np.sin(pos / denom)          # even columns -> sine
    PE[:, 1::2] = np.cos(pos / denom)          # odd  columns -> cosine
    return PE

seq_len, d_model = 50, 128
PE = positional_encoding(seq_len, d_model)

fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
im = ax.imshow(PE, aspect="auto", cmap="RdBu", interpolation="nearest")

ax.set_xlabel("embedding dimension (fast clocks --> slow clocks)", fontsize=13)
ax.set_ylabel("position in sequence", fontsize=13)
ax.set_title("Sinusoidal Positional Encoding",
             fontsize=15, pad=14, weight="bold")

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
cbar.set_label("value", rotation=270, labelpad=15)

ax.annotate("low dims flip fast\n(distinguish neighbors)",
            xy=(6, 40), xytext=(30, 44), fontsize=10,
            arrowprops=dict(arrowstyle="->", lw=1.2),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))
ax.annotate("high dims drift slow\n(distinguish far apart)",
            xy=(118, 10), xytext=(60, 4), fontsize=10,
            arrowprops=dict(arrowstyle="->", lw=1.2),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))

plt.tight_layout()
plt.show()