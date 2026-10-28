import torch

x = torch.randn((3, 129), dtype=torch.float32)
y = torch.randn((3, 1), dtype=torch.float32)
result = torch.relu(x + y) * y
