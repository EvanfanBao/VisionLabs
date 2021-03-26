import torchvision 
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from model2 import CNN
# 数据集加载
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010)),
])
testset = torchvision.datasets.CIFAR10(root='./data/', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset,batch_size=4,shuffle=False)

def main():
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    model = CNN()
    model.load_state_dict(torch.load('checkpoint/model_4.pth'))
     # 使用GPU
    if torch.cuda.is_available():
        model.cuda()
    data_len = len(testset)
    correct_num = 0
    for i, data in enumerate(testloader):
        inputs, labels = data
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()
        inputs, labels = Variable(inputs), Variable(labels)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        for j in range(len(predicted)):
            predicted_num = predicted[j].item()
            label_num = labels[j].item()
            if predicted_num == label_num:
                correct_num += 1
    
    correct_rate = correct_num / data_len
    print('correct rate is {:.3f}%'.format(correct_rate * 100))

if __name__ == "__main__":
    main()
        
