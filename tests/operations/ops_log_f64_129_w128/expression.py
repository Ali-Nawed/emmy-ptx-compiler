import torch

x = torch.randn((129,), dtype=torch.float64)
result = torch.log(x*x + 1)
