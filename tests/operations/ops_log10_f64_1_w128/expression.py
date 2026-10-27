import torch

x = torch.randn((1,), dtype=torch.float64)
result = torch.log(x*x + 1) / 2.302585092994046
