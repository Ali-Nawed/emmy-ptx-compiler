import torch

x = torch.randn((128,), dtype=torch.float32)
result = x * -1.25 + 0.5
