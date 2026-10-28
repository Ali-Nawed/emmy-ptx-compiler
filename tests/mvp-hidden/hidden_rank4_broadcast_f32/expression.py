import torch

x = torch.randn(2, 3, 1, 33)
y = torch.randn(1, 3, 5, 1)
result = torch.relu(x+y) * (x-y)
