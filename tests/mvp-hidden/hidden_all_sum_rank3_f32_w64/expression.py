import torch

x = torch.randn((2, 3, 33), dtype=torch.float32)
result = x.sum()
