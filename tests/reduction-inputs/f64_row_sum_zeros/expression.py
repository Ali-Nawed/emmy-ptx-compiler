import torch

x = torch.randn((3, 257), dtype=torch.float64)
result = x.sum(-1)
