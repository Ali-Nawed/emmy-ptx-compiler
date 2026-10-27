import torch

x = torch.randint(-3, 4, (1,), dtype=torch.int64)
y = torch.randint(-3, 4, (1,), dtype=torch.int64)
result = torch.pow(x, y*y)
