import torch

x = torch.randn((129,), dtype=torch.float32)
y = torch.randn((129,), dtype=torch.float32)
result = torch.log(x*x + 1) + torch.exp(y * 0.125) + torch.pow(x*x + 1, y * 0.125)
