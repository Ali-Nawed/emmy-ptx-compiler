import torch

x = torch.randn(129, dtype=torch.float32)
result = torch.sqrt(x * x + 1)
