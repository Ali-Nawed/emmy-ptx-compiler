import torch

x = torch.randn((3, 257), dtype=torch.float32)
result = x.amax(-1)
