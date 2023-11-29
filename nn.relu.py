import torch
import torchvision
from torch import nn
from torch.nn import ReLU, Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("./data_conv2d" , train=False, transform=torchvision.transforms.ToTensor()
                                       ,download=True)
dataloader = DataLoader (dataset, batch_size=64)

class sanqi(nn.Module):
    def __init__(self):
        super(sanqi, self).__init__()
        self.relu1 = ReLU()
        self.sigmold = Sigmoid()

    def forward(self, input):
        output = self.sigmold(input)
        return output

sanqi = sanqi()
writer = SummaryWriter("logs")
step = 0
for data in dataloader:
    imgs, targets = data
    output = sanqi(imgs)
    writer.add_images("sigmoid_input", imgs, global_step=step)

    writer.add_images("sigmoid_output", output, step)
    step +=1


writer.close()