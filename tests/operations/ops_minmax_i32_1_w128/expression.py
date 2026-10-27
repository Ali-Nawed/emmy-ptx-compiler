import torch

x = torch.randint(-3, 4, (1,), dtype=torch.int32)
y = torch.randint(-3, 4, (1,), dtype=torch.int32)
result = torch.maximum(x, y) - torch.minimum(x, y)
