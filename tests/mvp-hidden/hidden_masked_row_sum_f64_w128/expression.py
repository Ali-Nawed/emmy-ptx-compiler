import torch

x = torch.randn((3, 129), dtype=torch.float64)
y = torch.randn((3, 129), dtype=torch.float64)
result = torch.where(x > y, x-y, x+y).sum(-1)
