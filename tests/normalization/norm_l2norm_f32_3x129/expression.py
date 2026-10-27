import torch

x = torch.randn(3, 129)
result = x / torch.sqrt((x * x).sum(-1, keepdim=True))
