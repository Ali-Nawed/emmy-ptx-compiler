import torch

x = torch.randn(3, 257)
result = x - x.mean(-1, keepdim=True)
