import torch

x = torch.randn((256,), dtype=torch.float32)
result = x * -1.25 + 0.5
