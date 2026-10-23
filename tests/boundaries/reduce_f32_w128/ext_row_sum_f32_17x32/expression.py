import torch

x = torch.randn((17, 32), dtype=torch.float32)
result = x.sum(-1)
