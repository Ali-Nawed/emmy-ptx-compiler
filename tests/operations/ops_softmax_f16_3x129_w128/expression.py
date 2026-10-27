import torch

x = torch.randn((3, 129), dtype=torch.float16)
result = torch.softmax(x, dim=-1)
