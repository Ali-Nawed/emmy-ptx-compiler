import torch

x = torch.randn((3, 127), dtype=torch.float32)
result = x.sum(-1)
