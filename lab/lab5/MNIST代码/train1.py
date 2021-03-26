import torchvision as tv
import torchvision.transforms as transforms

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.autograd import Variable

from model1 import Net
import numpy as np

if __name__ == '__main__':
    # 定义对数据的预处理
    transform = transforms.Compose([
        transforms.ToTensor(),  # 转换为Tensor，并归一化至[0, 1]
    ])

    # 训练集
    trainset = tv.datasets.MNIST(
        root='../../MNIST_data',
        train=True,
        download=False,
        transform=transform
    )
    trainloader = DataLoader(
        dataset=trainset,
        batch_size=4,
        shuffle=True
    )

    # MNIST数据集中的十种标签
    classes = ('0', '1', '2', '3', '4',
               '5', '6', '7', '8', '9')

    # 创建网络模型
    net = Net()

    if torch.cuda.is_available():
        # 使用GPU
        net.cuda()
     
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=0.9)

    for epoch in range(5):
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            # 输入数据
            inputs, labels = data
            inputs = inputs.numpy()
            inputs =  np.pad(inputs, ((0,0),(0,0),(2,2),(2,2)), 'constant')
            inputs = torch.tensor(inputs)
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()
            inputs, labels = Variable(inputs), Variable(labels)

            # 梯度清0
            optimizer.zero_grad()

            # forward + backward
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            # 更新参数
            optimizer.step()

            # 打印log信息
            running_loss += loss.item()
            
            # 每2000个batch打印一次训练状态
            if i % 100 == 99:
                print('[{}/{}][{}/{}] loss: {:.3f}'.format(epoch + 1, 5, (i + 1) * 4, len(trainset), running_loss / 100))
                running_loss = 0.0
        
        # 保存参数文件
        torch.save(net.state_dict(), 'checkpoints/model_{}.pth'.format(epoch + 1))
        print('model_{}.pth saved'.format(epoch + 1))

    print('Finished Training')



