import torch

x = torch.randn(129, dtype=torch.float64)
result = x * torch.sigmoid(x)
