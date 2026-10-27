import torch

x = torch.randn(129, dtype=torch.float32)
result = x.to(torch.float64)
