import torch

x = torch.randn((1, 1025), dtype=torch.float32)
result = x.sum(-1)
