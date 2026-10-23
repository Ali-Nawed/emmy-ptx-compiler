import torch

x = torch.randn((1025,), dtype=torch.float32)
y = torch.randn((1025,), dtype=torch.float32)
result = x - y
