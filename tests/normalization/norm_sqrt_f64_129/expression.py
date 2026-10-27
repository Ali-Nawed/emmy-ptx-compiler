import torch

x = torch.randn(129, dtype=torch.float64)
result = torch.sqrt(x * x + 1)
