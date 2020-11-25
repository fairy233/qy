import torch
import torch.nn as nn

a = torch.Tensor([[[[1, 2, 3, 4]]]])
fn = nn.Conv2d(1, 2, 1, padding=1)
b = fn(a)
print(b)
