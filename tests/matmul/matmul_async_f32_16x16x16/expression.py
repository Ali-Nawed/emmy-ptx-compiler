import torch

a = torch.randn(16, 16, dtype=torch.float32)
b = torch.randn(16, 16, dtype=torch.float32)
result = a @ b
