import torch

x = torch.randint(-3, 4, (3, 129), dtype=torch.int32)
y = torch.randint(-3, 4, (3, 129), dtype=torch.int32)
result = torch.where(x > y, x-y, x+y).sum(-1)
