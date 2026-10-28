import torch

x = torch.randint(-3, 4, (3, 33), dtype=torch.int32)
y = torch.randint(-3, 4, (33,), dtype=torch.int64)
result = (x+y)*(x-y)
