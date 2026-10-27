import torch

x = torch.randn((1,), dtype=torch.float64)
y = torch.randn((1,), dtype=torch.float64)
result = torch.maximum(x, y) - torch.minimum(x, y)
