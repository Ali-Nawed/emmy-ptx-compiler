import torch

x = torch.randint(-8, 9, (33,), dtype=torch.int32)
result = x * 2 - 3
