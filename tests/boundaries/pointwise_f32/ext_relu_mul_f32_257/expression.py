import torch

x = torch.randn((257,), dtype=torch.float32)
y = torch.randn((257,), dtype=torch.float32)
result = torch.relu(x) * y
