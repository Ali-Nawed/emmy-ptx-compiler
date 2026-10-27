import torch

x = torch.tensor([0, 1, -2, -1, 0, 1, -1], dtype=torch.int64)
y = torch.tensor([1, 0, -1, -2, 0, -1, 1], dtype=torch.int64)
result = torch.where((x + 1152921504606846976) >= (y + 1152921504606846976), x, y)
