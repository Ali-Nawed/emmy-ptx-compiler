import torch

x = torch.randint(-3, 4, (3, 33), dtype=torch.int32)
y = torch.randint(-3, 4, (3, 33), dtype=torch.int32)
result = x.sum(-1) + y.sum(-1)
