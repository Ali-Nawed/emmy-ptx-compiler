import torch

x = torch.randn((3, 1025), dtype=torch.float16)
result = x.sum(-1)
