import torch

x = torch.randn((129,), dtype=torch.float16)
y = torch.randn((129,), dtype=torch.float16)
result = torch.relu(x) * y
