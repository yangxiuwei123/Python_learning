import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

#最大池化：保留图像特征但减少数据量
dataset = torchvision.datasets.CIFAR10("./data_conv2d" , train=False, transform=torchvision.transforms.ToTensor()
                                       ,download=True)
dataloader = DataLoader(dataset, batch_size=64)

class Sanqi(nn.Module):

    def __init__(self):
        super(Sanqi,self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=True)

    def forward(self, input):
        output = self.maxpool1(input)

        return output

sanqi = Sanqi()

writer = SummaryWriter("logs")
step = 0
for data in dataloader:
    imgs, targets = data
    output = sanqi(imgs)
    writer.add_images("maxpool2d_input", imgs, step)
    writer.add_images("maxpool2d_output", output, step)
    step +=1

writer.close()
