import torch

x = torch.randint(-3, 4, (3, 33), dtype=torch.int32)
y = torch.randn((33,), dtype=torch.float64)
result = (x+y)*(x-y)
