import torch

x = torch.randint(-8, 9, (129,), dtype=torch.int32)
y = torch.randint(-8, 9, (129,), dtype=torch.int32)
result = x - y
