import torch

x = torch.randn((3, 257), dtype=torch.float16)
result = x.sum(-1)
