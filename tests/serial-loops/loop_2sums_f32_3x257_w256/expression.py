import torch

x = torch.randn((3, 257), dtype=torch.float32)
y = torch.randn((3, 257), dtype=torch.float32)
result = x.sum(-1) + y.sum(-1)
