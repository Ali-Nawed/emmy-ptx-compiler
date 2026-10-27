import torch

x = torch.randn((33,), dtype=torch.float32)
result = torch.exp(x * 0.25)
