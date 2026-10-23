import torch

x = torch.randn((1, 257), dtype=torch.float32)
result = x.sum(-1)
