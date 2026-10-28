import torch

x = torch.randn(3, 33)
y = torch.randn(3, 33)
result = (x.sum(-1), y.mean(-1))
