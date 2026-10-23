import torch

x = torch.randint(-8, 9, (3, 129), dtype=torch.int64)
result = x.sum(-1)
