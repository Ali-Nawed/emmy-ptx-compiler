import torch

x = torch.randint(-8, 9, (129,), dtype=torch.int64)
result = ((x * 2 - 3) >> 0) & 255
