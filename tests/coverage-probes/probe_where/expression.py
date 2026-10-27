import torch

x = torch.randn(129)
y = torch.randn(129)
result = torch.where(x > 0, x, y)
