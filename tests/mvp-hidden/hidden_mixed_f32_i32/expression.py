import torch

x = torch.randn((3, 33), dtype=torch.float32)
y = torch.randint(-3, 4, (33,), dtype=torch.int32)
result = (x+y)*(x-y)
