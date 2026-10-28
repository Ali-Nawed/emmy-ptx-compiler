import torch

x = torch.randn((3, 129), dtype=torch.float32)
y = torch.randn((3, 129), dtype=torch.float32)
result = torch.where(x > y, x-y, x+y).sum(-1)
