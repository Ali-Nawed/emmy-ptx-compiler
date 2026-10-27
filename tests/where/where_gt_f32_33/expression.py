import torch

x = torch.randn(33, dtype=torch.float32)
y = torch.randn(33, dtype=torch.float32)
result = torch.where(x > y, x, y)
