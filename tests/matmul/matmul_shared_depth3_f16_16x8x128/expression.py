import torch

a = torch.randn(16, 128, dtype=torch.float16)
b = torch.randn(128, 8, dtype=torch.float16)
result = a @ b
