import torch

x = torch.randn((1,), dtype=torch.float64)
result = torch.rsqrt(x*x + 1)
