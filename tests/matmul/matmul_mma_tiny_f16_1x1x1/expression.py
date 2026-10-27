import torch

a = torch.randn(1, 1, dtype=torch.float16)
b = torch.randn(1, 1, dtype=torch.float16)
result = a @ b
