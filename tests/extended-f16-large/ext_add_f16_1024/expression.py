import torch

x = torch.randn((1024,), dtype=torch.float16)
y = torch.randn((1024,), dtype=torch.float16)
result = x + y
