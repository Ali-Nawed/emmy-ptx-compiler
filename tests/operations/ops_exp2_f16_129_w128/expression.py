import torch

x = torch.randn((129,), dtype=torch.float16)
result = torch.exp(x * 0.25 * 0.6931471805599453)
