import torch

x = torch.randn((3, 129), dtype=torch.float32)
y = torch.randn((3, 129), dtype=torch.float32)
result = ((x - y) / (y*y + 1)).sum(-1)
