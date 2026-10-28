import torch

x = torch.randn((2, 3, 129), dtype=torch.float32)
result = x.mean(-1)
