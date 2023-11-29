import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("../dataset",train=False, transform=torchvision.transforms.ToTensor(),download=True)

dataloader = DataLoader(dataset, batch_size=64)

class sanqi(nn.Module):
    def __init__(self):
        super(sanqi, self).__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x

loss = nn.CrossEntropyLoss()
sanqi = sanqi()
optim = torch.optim.SGD(sanqi.parameters(), lr=0.01)  #定义一个优化器
for epoch in range(20):
    for data in dataloader:
        imgs, targets = data
        outputs = sanqi(imgs)
        result_loss = loss(outputs, targets)
        optim.zero_grad()  #梯度清零
        result_loss.backward()
        optim.step()
