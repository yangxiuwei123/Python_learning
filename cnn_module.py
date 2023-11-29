import torch
from torch import nn


#继承torch.nn.module类
#初始化，super对父类调用他的初始化函数
#前向传播，输入x

class sanqi(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        output = input + 1
        return output

sanqi = sanqi()
x = torch.tensor(1.0)
output = sanqi(x)
print(output)