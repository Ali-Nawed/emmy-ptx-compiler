import torch

x = torch.randn((129,), dtype=torch.float64)
y = torch.randn((129,), dtype=torch.float64)
result = torch.maximum(x, y) - torch.minimum(x, y)
