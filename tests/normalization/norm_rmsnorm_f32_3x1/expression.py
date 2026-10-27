import torch

x = torch.randn(3, 1)
result = x / torch.sqrt((x * x).mean(-1, keepdim=True) + 1e-5)
