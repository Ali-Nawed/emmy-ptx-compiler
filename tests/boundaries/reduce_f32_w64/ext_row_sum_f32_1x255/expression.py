import torch

x = torch.randn((1, 255), dtype=torch.float32)
result = x.sum(-1)
