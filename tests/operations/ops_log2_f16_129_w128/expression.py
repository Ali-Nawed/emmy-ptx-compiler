import torch

x = torch.randn((129,), dtype=torch.float16)
result = torch.log(x*x + 1) / 0.6931471805599453
