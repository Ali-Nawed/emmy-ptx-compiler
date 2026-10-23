import torch

x = torch.randn((257,), dtype=torch.float16)
y = torch.randn((257,), dtype=torch.float16)
result = x * y
