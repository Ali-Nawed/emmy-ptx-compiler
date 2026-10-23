import torch

x = torch.randn((1, 127), dtype=torch.float32)
result = x.sum(-1)
