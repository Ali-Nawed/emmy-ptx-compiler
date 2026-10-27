import torch

x = torch.randn((129,), dtype=torch.float16)
result = torch.exp(x * 0.25)
