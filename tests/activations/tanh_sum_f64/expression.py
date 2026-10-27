import torch

x = torch.randn(3, 129, dtype=torch.float64)
result = torch.tanh(x).sum(-1)
