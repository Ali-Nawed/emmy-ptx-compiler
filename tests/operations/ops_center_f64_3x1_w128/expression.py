import torch

x = torch.randn((3, 1), dtype=torch.float64)
result = x - x.mean(-1, keepdim=True)
