import torch

x = torch.randn((1,), dtype=torch.float32)
y = torch.randn((1,), dtype=torch.float32)
result = torch.relu(x) * y
