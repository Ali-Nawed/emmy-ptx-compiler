import torch

x = torch.randn((129,), dtype=torch.float32)
result = torch.rsqrt(x*x + 1)
