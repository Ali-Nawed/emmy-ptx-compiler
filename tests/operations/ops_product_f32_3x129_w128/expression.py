import torch

x = torch.randn((3, 129), dtype=torch.float32)
result = (1 + x * 0.01).prod(-1)
