import torch

x = torch.randint(-3, 4, (3, 129), dtype=torch.int64)
y = torch.randint(-3, 4, (129,), dtype=torch.int64)
result = torch.relu(x + y) * y
