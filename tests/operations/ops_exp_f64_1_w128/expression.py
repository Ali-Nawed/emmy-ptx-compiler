import torch

x = torch.randn((1,), dtype=torch.float64)
result = torch.exp(x * 0.25)
