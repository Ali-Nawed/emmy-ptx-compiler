import torch

x = torch.randint(-3, 4, (3, 1), dtype=torch.int32)
result = x.amax(-1)
