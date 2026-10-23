import torch

x = torch.randn((1025,), dtype=torch.float16)
result = x.sum()
