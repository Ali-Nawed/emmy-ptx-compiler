import torch

x = torch.randn((128,), dtype=torch.float16)
y = torch.randn((128,), dtype=torch.float16)
result = x + y
