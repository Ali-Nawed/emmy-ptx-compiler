import torch

x = torch.randn((129,), dtype=torch.float64)
y = torch.randn((129,), dtype=torch.float64)
result = -torch.abs(x) + y
