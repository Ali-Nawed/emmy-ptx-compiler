import torch

x = torch.randn((1,), dtype=torch.float64)
y = torch.randn((1,), dtype=torch.float64)
result = torch.relu(x) * y
