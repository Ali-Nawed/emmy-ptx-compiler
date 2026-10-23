import torch

x = torch.randint(-8, 9, (1,), dtype=torch.int64)
result = x * 2 - 3
