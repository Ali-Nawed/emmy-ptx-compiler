import torch

a = torch.randn(16, 16, dtype=torch.float16)
b = torch.randn(8, 16, dtype=torch.float16)
result = a @ b.transpose(0, 1)
