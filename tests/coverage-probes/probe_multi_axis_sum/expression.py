import torch

x = torch.randn(2, 3, 129)
result = x.sum((0, 2))
