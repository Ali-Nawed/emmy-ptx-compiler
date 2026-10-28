import torch

x = torch.randn((3, 33), dtype=torch.float16)
y = torch.randint(-3, 4, (33,), dtype=torch.int64)
result = (x+y)*(x-y)
