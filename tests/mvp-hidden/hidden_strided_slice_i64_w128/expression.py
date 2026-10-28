import torch

x = torch.randint(-3, 4, (257,), dtype=torch.int64)
result = x[::2] + x[1::2].sum()
