import torch

x = torch.randint(-8, 9, (3, 1025), dtype=torch.int32)
result = x.sum(-1)
