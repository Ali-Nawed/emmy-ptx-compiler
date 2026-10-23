import torch

x = torch.randn((17, 31), dtype=torch.float32)
result = x.sum(-1)
