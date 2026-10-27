import torch

x = torch.randn(129, dtype=torch.float16)
result = torch.sqrt(x * x + 1)
