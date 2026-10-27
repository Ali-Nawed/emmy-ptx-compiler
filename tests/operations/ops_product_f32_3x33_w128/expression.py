import torch

x = torch.randn((3, 33), dtype=torch.float32)
result = (1 + x * 0.01).prod(-1)
