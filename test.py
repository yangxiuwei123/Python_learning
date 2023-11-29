import torch
import torchvision
from PIL import Image
from torch import nn

#模型验证
image_path = "imgs/airplan.png"
image = Image.open(image_path)
print(image)
image = image.convert('RGB')  #因为png图片是四个通道，除了RGB外还有一个透明度，调用convert保留其颜色通道，加上这步后可以适应各种格式图片，若本身就是三通道图片，加上此操作，不变

transform = torchvision.transforms.Compose(
    [torchvision.transforms.Resize((32,32)),
     torchvision.transforms.ToTensor()]
)

image = transform(image)
print(image.shape)

class Sanqi(nn.Module):
    def __init__(self):
        super(Sanqi, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),  # 展平后为64*4*4
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x

model = torch.load("sanqi_29_gpu.pth",map_location=torch.device('cpu'))  #GPU上的东西想要在cpu上运行，需要先映射
print(model)

image = torch.reshape(image,(1, 3, 32, 32))
model.eval()
with torch.no_grad():
    output = model(image)
print(output)
print(output.argmax(1))