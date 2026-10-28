import torch

x = torch.randn((2, 3, 33), dtype=torch.float64)
y = torch.randn((2, 3, 33), dtype=torch.float64)
result = torch.relu(x-y) * (x+y)
