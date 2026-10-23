import torch

x = torch.randint(-8, 9, (3, 33), dtype=torch.int32)
result = x.sum(-1)
