import torch

x = torch.randn(3, 31)
result = x - x.mean(-1, keepdim=True)
