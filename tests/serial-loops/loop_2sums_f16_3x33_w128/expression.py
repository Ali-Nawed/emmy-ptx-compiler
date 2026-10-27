import torch

x = torch.randn((3, 33), dtype=torch.float16)
y = torch.randn((3, 33), dtype=torch.float16)
result = x.sum(-1) + y.sum(-1)
