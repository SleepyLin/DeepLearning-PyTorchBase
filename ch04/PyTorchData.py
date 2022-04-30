# torch.utils.data
# dataset: __getitem_() \ _len_()
# dataloader: batch \ shuffle
import torch
from torch.utils import data
import numpy as np

# define specific dataset extends dataset abstract class
class TestDataset(data.Dataset):
    def __init__(self):
        self.Data = np.asarray([[1,2],[3,4],[2,1],[3,4],[4,5]])
        self.Label = np.asarray([0,1,0,1,2])

    def __getitem__(self, index):
        # transform numpy to tensor
        txt = torch.from_numpy(self.Data[index])
        label = torch.tensor(self.Label[index])
        return txt, label

    def __len__(self):
        return len(self.Data)
# 通过实例化数据集类，每次仅可通过调用__getitem()__获得一个样本
Test = TestDataset()
print(Test[2])
print(Test.__len__())
# 通过dataloader类，能够进行数据的批量处理、shuffle与并行加速
test_loader = data.DataLoader(Test, batch_size=2, shuffle=False, num_workers=2)
for i, traindata in enumerate(test_loader):
    print('i:', i)
    # 批量读取
    Data, Label = traindata
    print('data:', Data)
    print('Label:', Label)



# torchvision
import torchvision.transforms as transforms
# transforms:对PIL Image对象与Tensor对象操作
transforms.Compose([
    # 中心切割
    transforms.CenterCrop(10),
    # 随机选取切割中心点位置
    transforms.RandomCrop(20, padding=0),
    # 讲PIL.Image/numpy.ndarray数据转换为torch.FloatTensor
    transforms.ToTensor(),
    # 数据规范化[-1, 1]
    transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
])



from torchvision import datasets


# datasets
path = './'
# ImageFolder将path中的文件夹转换为序列
dataset_example = datasets.ImageFolder(path)
loader = data.DataLoader(dataset_example)

from torchvision import transforms, utils
import matplotlib.pyplot as plt

# define transform
my_trans = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])
# load data and transform
train_data = datasets.ImageFolder('./data/torchvision_data', transform = my_trans)
# 利用dataloader加载数据
train_loader = data.DataLoader(train_data, batch_size=8, shuffle = True)

for i_batch, img in enumerate(train_loader):
    # 打开index为0的数据进行展示
    if i_batch == 0:
        print(img[1])
        fig = plt.figure()
        grid = utils.make_grid(img[0])
        plt.imshow(grid.numpy().transpose((1, 2, 0)))
        plt.show()
        utils.save_image(grid, 'test01.png')
    break

# 可视化工具tensorboardX
# 安装：pip install tensorboardX
from tensorboardX import SummaryWriter
# 指明日志存放位置
writer = SummaryWriter(log_dir = 'logs')
# 调用示例add_xx()
writer.add_custom_scalars()
writer.close()

# 启动tensorboard服务：tensorboard --logdir=logs --port 6006
# http://localhost

# 利用tensorboardX可视化神经网络
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tensorboardX import SummaryWriter

# define network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        self.bn = nn.BatchNorm2d(20)

    def forward(self, x):
        x = F.max_pool2d(self.conv1(x), 2)
        x = F.relu(x) + F.relu(-x)
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = self.bn(x)
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x

input = torch.rand(32, 1, 28, 28)
model = Net()
# 将model保存为graph
with SummaryWriter(log_dir='logs', comment='Net') as w:
    w.add_graph(model, (input, ))

#　tensorboardX可视化损失值
writer = SummaryWriter(log_dir='logs', comment='Linear')
num_epochs = 100
for epoch in range(num_epochs):
    loss = epoch
    writer.add_scalar("loss", loss, epoch)

# tensorboardX可视化特征图
import torchvision.utils as vutils
writer = SummaryWriter(log_dir='logs', comment='feature map')
model.eval()
for name, layer in model._modules.items():
    if 'layer' in name or 'conv' in name:
        x = x.view(x.size(0), -1) if "fc" in name else x
        x1 = x.transpose(0, 1)
        img_grid = vutils.make_grid(x1, normalize=True, scale_each=True, nrow=4)  # n
        writer.add_image(f'{name}_feature_maps', img_grid, global_step=0)



