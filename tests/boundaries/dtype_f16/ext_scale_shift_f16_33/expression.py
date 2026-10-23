import torch

x = torch.randn((33,), dtype=torch.float16)
result = x * -1.25 + 0.5
