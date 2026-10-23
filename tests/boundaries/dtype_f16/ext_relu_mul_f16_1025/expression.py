import torch

x = torch.randn((1025,), dtype=torch.float16)
y = torch.randn((1025,), dtype=torch.float16)
result = torch.relu(x) * y
