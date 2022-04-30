import torch
import numpy as np
# tensor/ndarray: GPU/CPU

# create tensor
# list to tensor
print(torch.Tensor([1, 2, 3, 4, 5, 6]))
print(torch.Tensor(2,3))
# tensor / Tensor
# Tensor: dtype FloatTensor
t1 = torch.Tensor(1)
t2 = torch.tensor(1)
print("t1的值{},t1的数据类型{}".format(t1, t1.type()))
print("t2的值{},t2的数据类型{}".format(t2, t2.type()))
# 生成一个单位矩阵
torch.eye(2, 2)
# 自动生成全是0的矩阵
torch.zeros(2, 3)
# 根据规则生成数据
torch.linspace(1, 10, 4)
# 生成满足均匀分布随机数
torch.rand(2, 3)
# 生成满足标准分布随机数
torch.randn(2, 3)
# 返回所给数据形状相同，值全为0的张量
torch.zeros_like(torch.rand(2, 3))

# tensor reshape
# 生成一个形状为2x3的矩阵
x = torch.randn(2, 3)
# 查看矩阵的形状
print(x.size())
# 查看x的维度:2
print(x.dim())
# reshape x to 3x2
x.view(3, 2)
# reshape x to dimension 1
y = x.view(-1)
print(y.shape)
# 添加一个维度
z = torch.unsqueeze(y, 0)
# 查看z的形状
z.size()  # 结果为torch.Size([1, 6])
# 计算Z的元素个数
z.numel()  # 结果为6

# tensor index
x = torch.randn(2,3)
# get the first row
print(x[0, :])
# get the last line
print(x[:, -1])
# get the value > 0
mask = x>0
torch.masked_select(x, mask)

# broadcast
A = np.arange(0, 40, 10).reshape(4, 1)
B = np.arange(0, 3)
A1 = torch.from_numpy(A)  #形状为4x1
B1 = torch.from_numpy(B)  #形状为3
# similar to numpy
C = A1 + B1

# calc tensor
t = torch.randn(1, 3)
t1 = torch.randn(3, 1)
t2 = torch.randn(1, 3)
# t+0.1*(t1/t2)
torch.addcdiv(t, 0.1, t1, t2)
torch.sigmoid(t)
# t:[0,1]
torch.clamp(t, 0, 1)

# matrix calc
a = torch.tensor([2, 3])
b = torch.tensor([3, 4])
# 1D tensor dot produce
torch.dot(a, b)
#
x = torch.randint(10, (2, 3))
y = torch.randint(6, (3, 4))
# matrix multiplication
torch.mm(x, y)
x=torch.randint(10, (2, 2, 3))
y=torch.randint(6, (2, 3, 4))
# batch matrix multiplication
torch.bmm(x, y)

# autograd scalar
x=torch.Tensor([2])
# require grad
w=torch.randn(1,requires_grad=True)
b=torch.randn(1,requires_grad=True)
# no grad
x.detach()
torch.no_grad(x)
# forward
y = torch.mul(w, x)  # 等价于w*x
z = torch.add(y, b)  # 等价于y+b
# backward
z.backward()
# grad:tensor([2.]),tensor([1.]),None
print("参数w,b的梯度分别为:{},{},{}".format(w.grad,b.grad,x.grad))
print("非叶子节点y,z的梯度分别为:{},{}".format(y.grad,z.grad))

# autograd tensor
x = torch.tensor([[2, 3]], dtype=torch.float, requires_grad=True)
J = torch.zeros(2 ,2)
y = torch.zeros(1, 2)
y[0, 0] = x[0, 0] ** 2 + 3 * x[0 ,1]
y[0, 1] = x[0, 1] ** 2 + 2 * x[0, 0]
# y对x1的梯度
y.backward(torch.Tensor([[1, 0]]), retain_graph=True)
J[0] = x.grad
x.grad = torch.zeros_like(x.grad)
# y对x2的梯度
y.backward(torch.Tensor([[0, 1]]))
J[1] = x.grad