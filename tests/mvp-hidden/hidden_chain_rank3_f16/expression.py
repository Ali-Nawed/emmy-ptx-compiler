import torch

x = torch.randn((2, 3, 33), dtype=torch.float16)
y = torch.randn((2, 3, 33), dtype=torch.float16)
result = torch.relu(x-y) * (x+y)
