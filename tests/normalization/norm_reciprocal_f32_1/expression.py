import torch

x = torch.randn(1, dtype=torch.float32)
result = torch.reciprocal(x * x + 1)
