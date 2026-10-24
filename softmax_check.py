# function in your own words that takes a list of scores and returns the slices, using the same three steps:

# Raise e to each score.
# Add them up.
# Divide each number by the total.
import math
import torch


def softmax(scores):
    exps = [math.exp(score) for score in scores]
    total = sum(exps)
    return [exp / total for exp in exps]


if __name__ == "__main__":
    s = [1.0, 2.0, 3.0]
    # float64 on both sides, otherwise the comparison hides the real gap
    mine = torch.tensor(softmax(s), dtype=torch.float64)
    ref = torch.softmax(torch.tensor(s, dtype=torch.float64), dim=0)
    print(softmax(s))
    print((mine - ref).abs().max())