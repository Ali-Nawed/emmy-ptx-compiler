import torch

x = torch.randn((3, 129), dtype=torch.float64)
result = x.sum(-1)
