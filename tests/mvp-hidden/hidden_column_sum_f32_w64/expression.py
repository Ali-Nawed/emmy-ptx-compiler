import torch

x = torch.randn((33, 3), dtype=torch.float32)
result = x.sum(0)
