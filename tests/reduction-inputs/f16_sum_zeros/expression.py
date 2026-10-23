import torch

x = torch.randn((257,), dtype=torch.float16)
result = x.sum()
