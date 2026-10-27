import torch

x = torch.randn(129)
y = torch.randn(129)
result = torch.where(x > y, torch.where(x > 0, x, y), y)
