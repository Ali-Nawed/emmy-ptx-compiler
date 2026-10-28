import torch

x = torch.randint(-3, 4, (3, 33), dtype=torch.int32)
y = torch.randn((33,), dtype=torch.float16)
result = (x+y)*(x-y)
