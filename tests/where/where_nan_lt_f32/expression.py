import torch

x = torch.tensor([0.0, 1.0, -1.0, 2.0, -0.0], dtype=torch.float32)
y = torch.tensor([0.0, 1.0, 2.0, -1.0, 0.0], dtype=torch.float32)
result = torch.where((x / x) < y, y, -y)
