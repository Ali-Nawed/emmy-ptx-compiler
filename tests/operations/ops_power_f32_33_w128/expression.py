import torch

x = torch.randn((33,), dtype=torch.float32)
y = torch.randn((33,), dtype=torch.float32)
result = torch.pow(x*x + 1, y * 0.5)
