import torch

x = torch.randn(3, 1025)
result = x / torch.sqrt((x * x).sum(-1, keepdim=True))
