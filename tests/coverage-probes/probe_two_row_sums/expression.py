import torch

x = torch.randn(3, 129)
y = torch.randn(3, 129)
result = x.sum(-1) + y.sum(-1)
