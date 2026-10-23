import torch

x = torch.randn((1,), dtype=torch.float16)
result = x * -1.25 + 0.5
