import torch

x = torch.randn((17, 257), dtype=torch.float32)
result = x.sum(-1)
