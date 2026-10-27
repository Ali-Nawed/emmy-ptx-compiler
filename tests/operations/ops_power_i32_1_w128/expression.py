import torch

x = torch.randint(-3, 4, (1,), dtype=torch.int32)
y = torch.randint(-3, 4, (1,), dtype=torch.int32)
result = torch.pow(x, y*y)
