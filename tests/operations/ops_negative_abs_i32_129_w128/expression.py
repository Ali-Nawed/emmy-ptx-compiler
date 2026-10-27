import torch

x = torch.randint(-3, 4, (129,), dtype=torch.int32)
y = torch.randint(-3, 4, (129,), dtype=torch.int32)
result = -torch.abs(x) + y
