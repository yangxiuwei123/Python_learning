from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

"""
主要对图片进行一些变换,transforms.py文件是一个工具箱；里面有一些如totensor转换成tensor结构，resize等工具
对一个特定格式的图片经过其中的工具输出我们想要的一个图片结果
"""
#transforms的使用（1）
# python的用法 -> tensor数据类型
#通过transforms.ToTensor去解决两个问题：
#   1、transforms该如何使用（在Python中）
#   2、为什么我们需要Tensor数据类型

img_path = "C:\\Users\\HP\\Desktop\\数据集\\hymenoptera_data\\train\\bees_image\\1508176360_2972117c9d.jpg"
img = Image.open(img_path)

# 问题1：
#转换为tensor类型的图片
tensor_trans = transforms.ToTensor()  #相当于先实例化一个对象---tensor_trans
tensor_img = tensor_trans(img)  #调用上面实例化对象的__call__方法 ----把类当做函数使用的时候的定义

#常见的transforms
writer  = SummaryWriter("logs")
img = Image.open("C:\\Users\\HP\\Desktop\\数据集\\hymenoptera_data\\train\\bees_image\\1508176360_2972117c9d.jpg")

#(1)ToTensor
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
writer.add_image("ToTensor",img_tensor)

#(2)Normalize
"""
input[chanel] = (input[chanel] - mean[chanel]) / std[chanel]
(input-0.5) / 0.5 = 2 * input - 1   --->  input[0,1]  ---result--> result[-1,1]
"""
trans_norm = transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])  #均值和标准差，RGB三个通道所以每个参数三个数据
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image("Normalize", img_norm)

#(3-1)Resize
print(img.size)
trans_resize = transforms.Resize((512,512))
img_resize = trans_resize(img)  #resize的输入要为PIL的img
img_resize = trans_totensor(img_resize) #转换为tensor数据类型
print(img_resize)
writer.add_image("Resize",img_resize)
#(3-2)compose - resize
trans_resize_2 = transforms.Resize(300)
trans_compose = transforms.Compose([trans_resize_2, trans_totensor]) #相当于把上面的两步操作结合起来了
img_resize_2 = trans_compose(img) #img先传给trans_resize_2,得到的输出再输入到第二个trans_totensor，compose里有先后顺序
writer.add_image("resize_cmpose",img_resize_2)

#(4)RandomCrop
trans_random = transforms.RandomCrop((300,400))
trans_compose_2 = transforms.Compose([trans_random, trans_totensor])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("RandomCropHW",img_crop,i)

writer.close()

