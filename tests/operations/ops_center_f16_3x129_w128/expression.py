import torch

x = torch.randn((3, 129), dtype=torch.float16)
result = x - x.mean(-1, keepdim=True)
