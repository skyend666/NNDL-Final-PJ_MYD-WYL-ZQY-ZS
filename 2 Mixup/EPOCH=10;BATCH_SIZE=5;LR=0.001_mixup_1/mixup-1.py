import os
os.getcwd()
os.chdir("C:\\Users\\zhouqianyu\\Documents\\4. 研究生\\研一下-课程\\2 神经网络与深度学习\\期末pj\\test7")
print(os.getcwd())

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pickle
import random
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

  
"""
Import necessary libraries to train a network using mixup
The code is mainly developed using the PyTorch library
"""

"""
Dataset and Dataloader creation
All data are downloaded found via Graviti Open Dataset which links to CIFAR-10 official page
The dataset implementation is where mixup take place
"""

# 定义Summary_Writer
writer = SummaryWriter('./Result')   # 数据存放在这个文件夹
# cuda
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


"""
Define the hyperparameters, image transform components, and the dataset/dataloaders
"""
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# 定义全局变量
modelPath = './model.pkl'
batchSize = 5
nEpochs = 20
numPrint = 1000

#BATCH_SIZE = 64
NUM_WORKERS = 4
#LEARNING_RATE = 0.0001
#NUM_EPOCHS = 30

# 加载数据集 (训练集和测试集)
trainset = torchvision.datasets.CIFAR10(root='./Cifar-10', train=True, download=True, transform=transforms.ToTensor())
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchSize, shuffle=True)
testset = torchvision.datasets.CIFAR10(root='./Cifar-10', train=False, download=True, transform=transforms.ToTensor())
testloader = torch.utils.data.DataLoader(testset, batch_size=batchSize, shuffle=False)


# 定义神经网络  
class Net(nn.Module):  # 训练 ALexNet
    '''
    三层卷积，三层全连接  (应该是5层卷积，由于图片是 32 * 32，且为了效率，这里设成了 3 层)
    ''' 
    def __init__(self):
        super(Net, self).__init__()
        # 五个卷积层
        self.conv1 = nn.Sequential(  # 输入 32 * 32 * 3
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1),   # (32-3+2)/1+1 = 32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # (32-2)/2+1 = 16
        )
        self.conv2 = nn.Sequential(  # 输入 16 * 16 * 6
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, stride=1, padding=1),  # (16-3+2)/1+1 = 16
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # (16-2)/2+1 = 8
        )
        self.conv3 = nn.Sequential(  # 输入 8 * 8 * 16
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1), # (8-3+2)/1+1 = 8
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # (8-2)/2+1 = 4
        )
        self.conv4 = nn.Sequential(  # 输入 4 * 4 * 64
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1), # (4-3+2)/1+1 = 4
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # (4-2)/2+1 = 2
        )
        self.conv5 = nn.Sequential(  # 输入 2 * 2 * 128
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),# (2-3+2)/1+1 = 2
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # (2-2)/2+1 = 1
        )                            # 最后一层卷积层，输出 1 * 1 * 128
        # 全连接层
        self.dense = nn.Sequential(
            nn.Linear(128, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(-1, 128)
        x = self.dense(x)
        return x

net = Net().to(device)


# 使用测试数据测试网络
def Accuracy():
    correct = 0
    total = 0
    with torch.no_grad():  # 训练集中不需要反向传播
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device) # 将输入和目标在每一步都送入GPU
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)  # 返回每一行中最大值的那个元素，且返回其索引
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
    return 100.0 * correct / total

alpha = 0.2###
use_cuda = torch.cuda.is_available()


# 训练函数
def train():
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失
    #criterion = bceloss()#######################  
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)  # 随机梯度下降
    iter = 0
    num = 1
    
    # 训练网络
    for epoch in range(nEpochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            iter = iter + 1
            # 取数据
            images,target = data
            if use_cuda:
                images = images.cuda(non_blocking=True)
                target = torch.from_numpy(np.array(target)).float().cuda(non_blocking=True)
            images,target = Variable(images, volatile=True), Variable(target)
            # 2.mixup
            lam = np.random.beta(alpha,alpha)
            #randperm返回1~images.size(0)的一个随机排列
            if use_cuda:
                index = torch.randperm(images.size(0)).cuda()
            else:
                index = torch.randperm(images.size(0))
            #index = torch.randperm(images.size(0)).cuda()
            inputs = lam*images + (1-lam)*images[index,:]
            targets_a, targets_b = target, target[index]
            outputs = net(inputs)
            loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
            # 3.backward
            optimizer.zero_grad()   # reset gradient
            loss.backward()
            writer.add_scalar('loss', loss, iter)#######不是i是iter
            optimizer.step()  # 优化            
             # 统计数据
            running_loss += loss.item()
            if i % numPrint == 999:    # 每 batchsize * numPrint 张图片，打印一次
                print('epoch: %d\t batch: %d\t loss: %.6f' % (epoch + 1, i + 1, running_loss / (batchSize*numPrint)))
                running_loss = 0.0
                writer.add_scalar('accuracy', Accuracy(), num + 1)
                num = num + 1
    # 保存模型
    torch.save(net, './model.pkl')
    


    
if __name__ == '__main__':
    # 如果模型存在，加载模型
    if os.path.exists(modelPath):
        print('model exits')
        net = torch.load(modelPath)
        print('model loaded')
    else:
        print('model not exits')
    print('Training Started')
    train()
    writer.close()
    print('Training Finished')

