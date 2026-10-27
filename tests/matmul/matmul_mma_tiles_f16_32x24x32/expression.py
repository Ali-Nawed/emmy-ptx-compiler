import torch

a = torch.randn(32, 32, dtype=torch.float16)
b = torch.randn(32, 24, dtype=torch.float16)
result = a @ b
