import torch

x = torch.randn((256,), dtype=torch.float16)
y = torch.randn((256,), dtype=torch.float16)
result = x + y
