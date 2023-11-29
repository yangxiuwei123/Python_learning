
import  torch
import  torchvision

#保存方式1  ---》 加载模型
model1 = torch.load("vgg16_method1.pth")
print(model1)

#保存方式2  ---》 加载模型
#把保存的参数字典还原成vgg16模型
vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))

model2 = torch.load("vgg16_method2.pth")
print(model2)