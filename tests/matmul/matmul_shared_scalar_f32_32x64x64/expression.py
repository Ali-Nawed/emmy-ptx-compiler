import torch

a = torch.randn(32, 64, dtype=torch.float32)
b = torch.randn(64, 64, dtype=torch.float32)
result = a @ b
