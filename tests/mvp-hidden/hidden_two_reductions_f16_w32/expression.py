import torch

x = torch.randn((3, 129), dtype=torch.float16)
y = torch.randn((3, 129), dtype=torch.float16)
result = x.sum(-1) + y.sum(-1)
