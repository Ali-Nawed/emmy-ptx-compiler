import torch

x = torch.randn((129,), dtype=torch.float32)
y = torch.randn((129,), dtype=torch.float32)
result = -torch.abs(x) + y
