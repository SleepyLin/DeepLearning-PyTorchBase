import math
import numpy as np
from numpy import random as nr
"""
np.tab查看可用函数
np.abs查看详细信息
"""

# create ndarray

# list to ndarray
lst1 = [3.14, 2.17, 0, 1, 2]
lst2 = [[3.14, 2.17, 0, 1, 2], [1, 2, 3, 4, 5]]
nd1 =np.array(lst1)

#  random to ndarray
np.random.seed(123)              # 设置随机数种子
nd3 = np.random.random([3, 3])   # 0-1随机数
nd4 = np.random.randn(2, 3)      # 标准正态分布
np.random.shuffle(nd4)           # 打乱数据

# special ndarray
# 全0矩阵
nd5 = np.zeros([3, 3])
# 全1矩阵
nd6 = np.ones([3, 3])
# 单位矩阵
nd7 = np.eye(3)
# 对角矩阵
nd8 = np.diag([1, 2, 3])

# save ndarray
np.savetxt(X=nd1, fname='./test1.txt')
# load naarray
nd10 = np.loadtxt('./test1.txt')

# arrange(start:1 end:3.5 step:0.5)
print(np.arange(1, 4, 0.5))
# linspace(start, stop, num)
print(np.linspace(0, 1, 10))


# read ndarray
# 一维数组
nd11 = np.random.random([10])
# 取固定间隔数据
print(nd11[1:6:2])
# 倒序取数
print(nd11[::-2])

# 多维数组
nd12=np.arange(25).reshape([5,5])
# 取特定区域:2,3行
print(nd12[1:3,:])
print(nd12[[1,2]])
# 按间隔取数:第三行后间隔2，列数间隔2
print(nd12[2::2,::2])
# 取符合条件值
print(nd12[(nd12>3)&(nd12<10)])

# choice抽样
a=np.arange(1,25,dtype=float)
# size:输出数组形状 replace: True为重复抽取, p为各元素抽取概率
c2=nr.choice(a,size=(3,4),replace=False, p=a / np.sum(a))


# numpy
A = np.array([[1, 2], [-1, 4]])
B = np.array([[2, 0], [3, 4]])
# element-wise product
print(A*B)
print(np.multiply(A,B))
# dot product
print(np.dot(A, B))

# transform
arr = np.arange(10)
# row 2 line 5
print(arr.reshape(2, 5))
print(arr.reshape(-1, 5))
print(arr.resize(2, 5))
# transposition
print(arr.T)

# flat
arr1 = np.arange(6).reshape(2, -1)
# ravel 按列元素逐个展平
print(arr1.ravel('F'))
# flatten 按行元素展平,化为向量
print(arr1.flatten())

# reduce dimension
# 去除数值为1的维度
arr2 = np.arange(6).reshape(1, 6)
print(arr2.squeeze().shape)
# 维度交换 0,1,2 - 1,2,0
arr3 = np.arange(24).reshape(2, 3, 4)
print(arr3.transpose(1, 2, 0).shape)

# merge two array
a =np.arange(4).reshape(2, 2)
b = np.arange(4).reshape(2, 2)
# 行合并
c = np.append(a, b, axis=0)
c = np.concatenate((a, b), axis=0)
# 列合并
d = np.append(a, b, axis=1)
# 堆叠数组
print(np.stack((a, b), axis=0))

# batch processing
data_train = np.random.randn(10000,2,3)
np.random.shuffle(data_train)
# 定义批量大小
batch_size=100
# 进行批处理:利用切片进行批处理
for i in range(0, len(data_train), batch_size):
    x_batch_sum = np.sum(data_train[i:i+batch_size])
    print("第{}批次,该批次的数据之和:{}".format(i, x_batch_sum))

# numpy ufunc
x = [i * 0.001 for i in np.arange(1000000)]
# numpy sin
np.sin(x)
# math sin
for i, t in enumerate(x):
    x[i] = math.sin(t)

# broadcast
A = np.arange(0, 40, 10).reshape(4, 1)
B = np.arange(0, 3)
print("A矩阵的形状:{},B矩阵的形状:{}".format(A.shape, B.shape))
# B变为1*3, 而后取各维度最大值4*3，通过自我复制，A与B分别得到4*3
C = A+B
print("C矩阵的形状:{}".format(C.shape))
print(C)

