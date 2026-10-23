import torch

x = torch.randint(-8, 9, (129,), dtype=torch.int64)
y = torch.randint(-8, 9, (129,), dtype=torch.int64)
result = ((x * y) >> 0) & 255
