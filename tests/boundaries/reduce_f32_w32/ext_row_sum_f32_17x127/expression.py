import torch

x = torch.randn((17, 127), dtype=torch.float32)
result = x.sum(-1)
