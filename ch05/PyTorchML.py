import torch
import torch.nn as nn
# 过拟合与欠拟合
# 正则化
# dropout
net1_dropped = torch.nn.Sequential(
    torch.nn.Linear(13, 16),
    torch.nn.Dropout(0.5),  # drop 50% of the neuron
    torch.nn.ReLU(),
    torch.nn.Linear(16, 32),
    torch.nn.Dropout(0.5),  # drop 50% of the neuron
    torch.nn.ReLU(),
    torch.nn.Linear(32, 1),
)
# batch normalization
net1_nb = torch.nn.Sequential(
    torch.nn.Linear(13, 16),
    nn.BatchNorm1d(num_features=16),
    torch.nn.ReLU(),
    torch.nn.Linear(16, 32),
    nn.BatchNorm1d(num_features=32),
    torch.nn.ReLU(),
    torch.nn.Linear(32, 1),
)
#　权值初始化
## xavier: sigmoid \ tanh
## kaiming:relu

# 损失函数
## 回归：torch.nn.MSELoss()
## 分类：torch.nn.CrossEntropyLoss()

# 优化器
## 传统反向传播算法: 学习率敏感、鞍点问题
## 动量算法: 历史梯度与当前梯度合并
## AdaGrad: 自动调整学习率
## RMSProp: 针对非凸背景进行优化
## Adam: RMSProp+动量

# GPU加速
# 查看可用GPU数量
torch.cuda.device_count()
# 数据转移至GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Net()
device_ids =[0,1,2,3]
#对数据
input_data=input_data.to(device=device_ids[0])
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model.to(device)


