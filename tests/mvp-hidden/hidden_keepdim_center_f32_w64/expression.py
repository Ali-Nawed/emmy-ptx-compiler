import torch

x = torch.randn(3, 32)
result = x - x.mean(-1, keepdim=True)
