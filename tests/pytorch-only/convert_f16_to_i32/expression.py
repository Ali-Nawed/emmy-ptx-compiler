import torch

x = torch.randn(129, dtype=torch.float16)
result = x.to(torch.int32)
