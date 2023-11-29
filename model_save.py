import torch
import torchvision

vgg16 = torchvision.models.vgg16(pretrained = False)

#save 1
torch.save(vgg16, "vgg16_method1.pth")

#save 2 (官方推荐)
torch.save(vgg16.state_dict(), "vgg16_method2.pth")  #保存模型中的参数为字典