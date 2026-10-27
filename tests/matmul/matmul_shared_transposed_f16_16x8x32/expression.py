import torch

a = torch.randn(16, 32, dtype=torch.float16)
b = torch.randn(8, 32, dtype=torch.float16)
result = a @ b.transpose(0, 1)
