import torch

x = torch.randn((3, 129), dtype=torch.float16)
y = torch.randn((3, 1), dtype=torch.float16)
result = torch.relu(x + y) * y
