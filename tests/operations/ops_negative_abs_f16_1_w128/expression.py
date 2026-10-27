import torch

x = torch.randn((1,), dtype=torch.float16)
y = torch.randn((1,), dtype=torch.float16)
result = -torch.abs(x) + y
