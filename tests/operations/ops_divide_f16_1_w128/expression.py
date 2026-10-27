import torch

x = torch.randn((1,), dtype=torch.float16)
y = torch.randn((1,), dtype=torch.float16)
result = (x - y) / (y*y + 1)
