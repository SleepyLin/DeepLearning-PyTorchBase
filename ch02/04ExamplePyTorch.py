import torch
from matplotlib import pyplot as plt

# generate train dataset
torch.manual_seed(100)
dtype = torch.float
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
print(x)
y = 3*x.pow(2) + 2 + 0.2*torch.rand(x.size())

w = torch.randn(1, 1, dtype=dtype, requires_grad=True)
b = torch.randn(1, 1, dtype=dtype, requires_grad=True)

lr = 0.001
for i in range(800):
    # forward
    y_pred = x.pow(2).mm(w) + b
    # loss
    loss = 0.5*(y_pred-y)**2
    loss = loss.sum()
    #backward
    loss.backward()
    #更新参数
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad

        w.grad.zero_()
        b.grad.zero_()
plt.plot(x.numpy(), y_pred.detach().numpy(), 'r-', label='predict')  # predict
plt.scatter(x.numpy(), y.numpy(), color='blue', marker='o', label='true')  # true data
plt.xlim(-1, 1)
plt.ylim(2, 6)
plt.legend()
plt.show()

print(w, b)