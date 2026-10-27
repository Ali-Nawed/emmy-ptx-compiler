import torch

x = torch.randn(129, dtype=torch.float16)
result = torch.reciprocal(x * x + 1)
