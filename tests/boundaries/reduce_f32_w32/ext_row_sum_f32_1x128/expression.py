import torch

x = torch.randn((1, 128), dtype=torch.float32)
result = x.sum(-1)
