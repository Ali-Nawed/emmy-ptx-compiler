import torch

a = torch.randn(17, 32, dtype=torch.float16)
b = torch.randn(32, 13, dtype=torch.float16)
result = a @ b
