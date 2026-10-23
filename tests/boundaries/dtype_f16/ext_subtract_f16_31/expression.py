import torch

x = torch.randn((31,), dtype=torch.float16)
y = torch.randn((31,), dtype=torch.float16)
result = x - y
