import torch

a = torch.randn(17, 19, dtype=torch.float16)
b = torch.randn(19, 13, dtype=torch.float16)
result = a @ b
