import torch

x = torch.randn(128, dtype=torch.float16)
result = torch.tanh(x) * 1.5
