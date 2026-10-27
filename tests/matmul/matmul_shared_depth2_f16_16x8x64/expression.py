import torch

a = torch.randn(16, 64, dtype=torch.float16)
b = torch.randn(64, 8, dtype=torch.float16)
result = a @ b
