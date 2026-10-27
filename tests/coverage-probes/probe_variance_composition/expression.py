import torch

x = torch.randn(3, 129)
result = (x*x).mean(-1) - x.mean(-1)**2
