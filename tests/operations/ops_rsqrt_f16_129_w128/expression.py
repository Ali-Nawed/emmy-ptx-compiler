import torch

x = torch.randn((129,), dtype=torch.float16)
result = torch.rsqrt(x*x + 1)
