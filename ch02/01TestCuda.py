import torch
from torch.backends import cudnn
#test CUDA
print("Support CUDA ?: ", torch.cuda.is_available())
x = torch.tensor([10.0])
x = x.cuda()
print(x)
y = torch.randn(2, 3)
y = y.cuda()
print(y)
z = x + y
print(z)

# test CUDNN
print("Support cudnn ?: ", cudnn.is_acceptable(x))

# 2.4 tensor introduction
