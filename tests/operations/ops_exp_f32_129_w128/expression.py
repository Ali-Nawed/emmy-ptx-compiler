import torch

x = torch.randn((129,), dtype=torch.float32)
result = torch.exp(x * 0.25)
