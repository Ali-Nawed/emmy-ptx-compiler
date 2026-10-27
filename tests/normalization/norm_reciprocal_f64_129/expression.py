import torch

x = torch.randn(129, dtype=torch.float64)
result = torch.reciprocal(x * x + 1)
