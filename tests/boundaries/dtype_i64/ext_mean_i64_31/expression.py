import torch

x = torch.randint(-8, 9, (31,), dtype=torch.int64)
result = x.mean(dtype=torch.float32)
