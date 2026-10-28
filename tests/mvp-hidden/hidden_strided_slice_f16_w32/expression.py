import torch

x = torch.randn((257,), dtype=torch.float16)
result = x[::2] + x[1::2].sum()
