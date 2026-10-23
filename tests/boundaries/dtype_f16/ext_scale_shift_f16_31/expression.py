import torch

x = torch.randn((31,), dtype=torch.float16)
result = x * -1.25 + 0.5
