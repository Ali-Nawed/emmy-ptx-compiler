import torch

x = torch.randn((129,), dtype=torch.float16)
y = torch.randn((129,), dtype=torch.float16)
result = torch.maximum(x, y) - torch.minimum(x, y)
