import torch

x = torch.randn((3, 33), dtype=torch.float32)
result = torch.softmax(x, dim=-1)
