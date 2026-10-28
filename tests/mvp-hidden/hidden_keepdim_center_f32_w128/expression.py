import torch

x = torch.randn(3, 33)
result = x - x.mean(-1, keepdim=True)
