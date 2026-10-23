import torch

x = torch.randint(-8, 9, (257,), dtype=torch.int32)
y = torch.randint(-8, 9, (257,), dtype=torch.int32)
result = x - y
