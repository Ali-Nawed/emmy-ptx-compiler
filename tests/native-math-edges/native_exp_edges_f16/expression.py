import torch

x = torch.tensor([-17.375, -17.34375, -17.328125, -17.3125, -17.28125, -16.0, -10.0, 0.0, 10.0, 11.0, 11.078125, 11.0859375], dtype=torch.float16)
result = torch.exp(x)
