# 下学期第一次任务

##### 复现需要将device改为cuda，本机使用mps

## 一.  MINIST手写数据集识别

### 使用最弱的全连接网络进行训练

使用pytorch自带的minist数据集API进行对于数据集的下载

~~开袋即食~~

```python
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# 创建数据加载器
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
```

对于全局变量的定义

```python
# 2. 定义超参数
input_size = 784  # MNIST图片大小是28x28
hidden_sizes = [128, 64]  # 隐藏层的大小
output_size = 10  # 输出的类别数为10，分别对应0到9的数字
num_epochs = 50  # 进行5次训练迭代
batch_size = 64  # 每批次处理64张图片
learning_rate = 0.001  # 学习率设置为0.001

# 3. 数据准备
transform = transforms.Compose([
    transforms.ToTensor(),  # 将图片转换成PyTorch的Tensor格式
    transforms.Normalize((0.5,), (0.5,))  # 标准化处理，以减小模型对数据规模的敏感性
])
```
对于神经网络定义的部分，直接暴力全连接三层

```python
 def __init__(self, input_size, hidden_sizes, output_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], output_size)
        self.relu = nn.ReLU()
```
传播函数使用ReLU激活函数进行激活

```python
def forward(self, x):
    x = self.relu(self.fc1(x))
    x = self.relu(self.fc2(x))
    x = self.fc3(x)
    return x
```

训练的效果似乎是不差的，但是问题出在了OJ上评测的时候似乎model_load_error

模型保存的问题？但是接下来使用的cnn网络没有问题

### 使用CNN神经网络进行图像的分析

```python
def __init__(self):
        super(Net, self).__init__()
        # 卷积层
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.batchnorm3 = nn.BatchNorm2d(128)
        # Dropout层
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        # 全连接层
        self.fc1 = nn.Linear(128 * 7 * 7, 256) # 更正全连接层输入特征数
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.batchnorm1(self.conv1(x)))
        x = F.relu(self.batchnorm2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.batchnorm3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
```

这段代码定义了一个具有多个卷积层、批量归一化层和全连接层的卷积神经网络

`self.conv1` 第一个卷积层，使用一个单通道的输入，输出32个特征映射，并使用大小为3x3的卷积核和1的填充，以确保在卷积操作后输出的维度不变。

`self.batchnorm1` 第一个批量归一化层，它对应于卷积层conv1的输出。批量归一化可以加速训练过程，并有助于防止过拟合。

`self.dropout1`和`self.dropout2`是Dropout层，这两层分别以0.25和0.50的概率暂时丢弃一部分特征，以减少模型过拟合。

最后添加两个全连接层，看起来似乎可以增加对于MINIST数据集识别的准确度

对，`看起来` `似乎`

鉴于我现在对于各种机器学习的结构仅保留在把不同功能的“积木”搭起来的阶段。

所以这个网络看起来挺好，表现也就一般。

并且问题出在了这个网络会随着训练的epoch增加分数降低。。。

😭

## YOLO检测机器人