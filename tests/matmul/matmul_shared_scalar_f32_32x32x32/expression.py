import torch

a = torch.randn(32, 32, dtype=torch.float32)
b = torch.randn(32, 32, dtype=torch.float32)
result = a @ b
