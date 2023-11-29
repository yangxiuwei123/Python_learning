import torch
import torchvision
from torch import nn

# vgg16_true = torchvision.models.vgg16(pretrained = True)
vgg16_false = torchvision.models.vgg16(pretrained = False)

train_data = torchvision.datasets.CIFAR10("../dataset",train=True, transform=torchvision.transforms.ToTensor(),download=True)
print(vgg16_false)

vgg16_false.add_module('add_linear',nn.Linear(1000, 10))  #给VGG16再加一层linear层
vgg16_false.classifier.add_module('add_linear',nn.Linear(1000, 10))  #给VGG16中的classifier里面再加一层linear层

print(vgg16_false)


#修改vgg16(没有添加的原始模型的最后一层)的最后一层
vgg16_false.classifier[6] = nn.Linear(4096, 10)
print(vgg16_false)