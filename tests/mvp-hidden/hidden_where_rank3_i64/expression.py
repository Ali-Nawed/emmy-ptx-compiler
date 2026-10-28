import torch

x = torch.randint(-3, 4, (2, 3, 33), dtype=torch.int64)
y = torch.randint(-3, 4, (2, 3, 33), dtype=torch.int64)
result = torch.where(x > y, x-y, x+y)
