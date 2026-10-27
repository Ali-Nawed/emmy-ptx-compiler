import torch

x = torch.randn(33, dtype=torch.float16)
result = torch.tanh(x)
