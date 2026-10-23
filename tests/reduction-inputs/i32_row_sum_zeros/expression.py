import torch

x = torch.randint(-8, 9, (3, 257), dtype=torch.int32)
result = x.sum(-1)
