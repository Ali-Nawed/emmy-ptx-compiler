import torch

x = torch.randn((3, 33), dtype=torch.float16)
y = torch.randn((33,), dtype=torch.float32)
result = (x+y)*(x-y)
