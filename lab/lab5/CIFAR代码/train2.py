import torch 
import torchvision 
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from model2 import CNN
from torch.autograd import Variable
# 数据集加载
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010)),
])
trainset = torchvision.datasets.CIFAR10(root='./data/', train = True, download=False, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True)


net = CNN()
# 使用GPU
if torch.cuda.is_available():
    net.cuda()
# 损失函数
criterion = nn.CrossEntropyLoss()
# 优化器
optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=0.9)

for epoch in range(4):
    running_loss = 0.0
    for i, data in enumerate(trainloader,0):
        inputs, labels = data
        # GPU
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()
        # 封装为Variable
        inputs, labels = Variable(inputs), Variable(labels)
        # 梯度清0
        optimizer.zero_grad()
        
        # forward
#        print("enter forward")
        outputs = net(inputs) #net
        # 计算损失函数
#        print("enter loss calc")
        loss = criterion(outputs, labels)
        # 反向传播
#        print("enterloss.backward")
        loss.backward()
#        print("enter back propagation")
        optimizer.step()
        
        
        running_loss += loss.item()
        # 每100个batch打印一次训练状态
        if i % 10 == 9:
            print('[{}/{}][{}/{}] loss: {:.3f}'.format(epoch + 1, 4, (i + 1) * 4, len(trainset), running_loss / 10))
            running_loss = 0
        #if i % 100 == 99:
        #    print('[{}/{}][{}/{}] loss: {:.3f}'.format(epoch + 1, 5, (i + 1) * 4, len(trainset), running_loss / 100))
        #    running_loss = 0


    print('==> Saving model ...')
    state = {
        'net': net,
        'epoch': epoch,
    }
    
    
    torch.save(net.state_dict(),'./checkpoint/model_{}.pth'.format(epoch+1))
    print('model_{}.pth saved'.format(epoch+1))

print('==> Finished Training ...')
