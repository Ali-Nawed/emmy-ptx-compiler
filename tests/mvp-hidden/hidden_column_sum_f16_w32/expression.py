import torch

x = torch.randn((33, 3), dtype=torch.float16)
result = x.sum(0)
