import torch

x = torch.randint(-3, 4, (3, 129), dtype=torch.int64)
y = torch.randint(-3, 4, (3, 129), dtype=torch.int64)
result = torch.where(x > y, x-y, x+y).sum(-1)
