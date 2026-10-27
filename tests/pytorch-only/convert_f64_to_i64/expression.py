import torch

x = torch.randn(129, dtype=torch.float64)
result = x.to(torch.int64)
