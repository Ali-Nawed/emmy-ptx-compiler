import torch

x = torch.randint(-3, 4, (2, 3, 33), dtype=torch.int32)
result = x.sum()
