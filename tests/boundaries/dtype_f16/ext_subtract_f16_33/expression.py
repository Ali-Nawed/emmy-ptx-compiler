import torch

x = torch.randn((33,), dtype=torch.float16)
y = torch.randn((33,), dtype=torch.float16)
result = x - y
