import numpy as np
from matplotlib import pyplot as plt

# generate input x and target y
np.random.seed(100)
x = np.linspace(-1, 1, 100).reshape(100, 1)
y = 3 * np.power(x, 2) + 2 + 0.2 * np.random.rand(x.size).reshape(100, 1)
# plot x and y
plt.scatter(x, y)
plt.show()

# initiate w and b
w1 = np.random.rand(1, 1)
b1 = np.random.rand(1, 1)

# loss backward
lr = 0.001
for i in range(800):
    # forward:prediction
    y_pred = np.power(x, 2) * w1 + b1
    # loss function
    loss = 0.5 * (y_pred - y) ** 2
    loss = loss.sum
    # calc grad
    grad_w = np.sum(np.power(x, 2)*(y_pred-y))
    grad_b = np.sum(y_pred-y)
    w1 -= lr*grad_w
    b1 -= lr*grad_b
plt.plot(x, y_pred,'r-',label='predict')
plt.scatter(x, y,color='blue',marker='o',label='true') # true data
plt.xlim(-1,1)
plt.ylim(2,6)
plt.legend()
plt.show()
print(w1,b1)
