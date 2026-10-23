import torch

x = torch.randn((17, 1025), dtype=torch.float32)
result = x.sum(-1)
