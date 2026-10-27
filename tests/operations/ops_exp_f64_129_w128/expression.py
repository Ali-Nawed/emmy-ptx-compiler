import torch

x = torch.randn((129,), dtype=torch.float64)
result = torch.exp(x * 0.25)
