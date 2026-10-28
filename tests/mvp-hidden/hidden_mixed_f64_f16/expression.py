import torch

x = torch.randn((3, 33), dtype=torch.float64)
y = torch.randn((33,), dtype=torch.float16)
result = (x+y)*(x-y)
