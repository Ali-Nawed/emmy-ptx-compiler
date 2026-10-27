import torch

x = torch.randn(129, dtype=torch.float16)
y = torch.randn(129, dtype=torch.float16)
result = torch.where(x > 0, x, y)
