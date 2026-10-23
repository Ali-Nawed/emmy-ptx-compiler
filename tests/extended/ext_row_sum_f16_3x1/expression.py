import torch

x = torch.randn((3, 1), dtype=torch.float16)
result = x.sum(-1)
