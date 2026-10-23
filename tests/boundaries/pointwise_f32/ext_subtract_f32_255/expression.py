import torch

x = torch.randn((255,), dtype=torch.float32)
y = torch.randn((255,), dtype=torch.float32)
result = x - y
