import torch

x = torch.randn((33,), dtype=torch.float32)
result = torch.log(x*x + 1)
