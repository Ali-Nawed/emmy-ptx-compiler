import torch

x = torch.randn((1,), dtype=torch.float64)
y = torch.randn((1,), dtype=torch.float64)
result = (x - y) / (y*y + 1)
