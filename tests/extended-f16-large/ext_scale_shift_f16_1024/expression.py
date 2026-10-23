import torch

x = torch.randn((1024,), dtype=torch.float16)
result = x * -1.25 + 0.5
