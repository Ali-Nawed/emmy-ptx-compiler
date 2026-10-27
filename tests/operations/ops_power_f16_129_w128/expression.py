import torch

x = torch.randn((129,), dtype=torch.float16)
y = torch.randn((129,), dtype=torch.float16)
result = torch.pow(x*x + 1, y * 0.5)
