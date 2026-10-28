import torch

x = torch.randint(-3, 4, (2, 3, 33), dtype=torch.int32)
y = torch.randint(-3, 4, (2, 3, 33), dtype=torch.int32)
result = torch.relu(x-y) * (x+y)
