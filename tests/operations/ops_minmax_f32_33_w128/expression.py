import torch

x = torch.randn((33,), dtype=torch.float32)
y = torch.randn((33,), dtype=torch.float32)
result = torch.maximum(x, y) - torch.minimum(x, y)
