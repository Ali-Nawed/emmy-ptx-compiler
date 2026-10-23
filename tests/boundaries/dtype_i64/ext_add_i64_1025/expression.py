import torch

x = torch.randint(-8, 9, (1025,), dtype=torch.int64)
y = torch.randint(-8, 9, (1025,), dtype=torch.int64)
result = x + y
