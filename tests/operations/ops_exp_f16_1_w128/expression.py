import torch

x = torch.randn((1,), dtype=torch.float16)
result = torch.exp(x * 0.25)
