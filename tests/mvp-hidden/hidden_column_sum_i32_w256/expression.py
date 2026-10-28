import torch

x = torch.randint(-3, 4, (33, 3), dtype=torch.int32)
result = x.sum(0)
