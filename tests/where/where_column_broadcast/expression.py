import torch

x = torch.randn(3, 129)
y = torch.randn((3, 1))
result = torch.where(x > y, x, y)
