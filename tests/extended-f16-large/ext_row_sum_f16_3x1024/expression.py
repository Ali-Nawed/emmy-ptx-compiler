import torch

x = torch.randn((3, 1024), dtype=torch.float16)
result = x.sum(-1)
