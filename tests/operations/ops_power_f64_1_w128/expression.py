import torch

x = torch.randn((1,), dtype=torch.float64)
y = torch.randn((1,), dtype=torch.float64)
result = torch.pow(x*x + 1, y * 0.5)
