import torch

x = torch.randint(-8, 9, (33,), dtype=torch.int64)
result = x.sum()
