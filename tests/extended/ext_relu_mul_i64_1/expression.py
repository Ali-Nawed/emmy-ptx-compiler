import torch

x = torch.randint(-8, 9, (1,), dtype=torch.int64)
y = torch.randint(-8, 9, (1,), dtype=torch.int64)
result = torch.relu(x) * y
