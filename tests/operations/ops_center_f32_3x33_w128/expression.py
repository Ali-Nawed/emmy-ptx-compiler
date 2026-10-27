import torch

x = torch.randn((3, 33), dtype=torch.float32)
result = x - x.mean(-1, keepdim=True)
