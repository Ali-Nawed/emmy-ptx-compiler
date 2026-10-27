import torch

x = torch.randn(129, dtype=torch.float32)
result = x * torch.sigmoid(x)
