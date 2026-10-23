import torch

x = torch.randn((3, 1025), dtype=torch.float32)
result = x.sum(-1)
