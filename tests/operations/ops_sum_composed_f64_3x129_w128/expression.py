import torch

x = torch.randn((3, 129), dtype=torch.float64)
y = torch.randn((3, 129), dtype=torch.float64)
result = ((x - y) / (y*y + 1)).sum(-1)
