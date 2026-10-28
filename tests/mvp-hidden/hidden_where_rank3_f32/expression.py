import torch

x = torch.randn((2, 3, 33), dtype=torch.float32)
y = torch.randn((2, 3, 33), dtype=torch.float32)
result = torch.where(x > y, x-y, x+y)
