import torch

x = torch.randn((17, 255), dtype=torch.float32)
result = x.sum(-1)
