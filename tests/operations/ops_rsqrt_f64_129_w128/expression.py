import torch

x = torch.randn((129,), dtype=torch.float64)
result = torch.rsqrt(x*x + 1)
