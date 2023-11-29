from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

"""
scalar_value:对应图像Y轴
global_step:对应X轴
"""
#tensorBoard的使用（2）
writer = SummaryWriter("logs")
# logs文件的使用，在终端使用命令tensorboard --logdir=computer_vision\pytorch\logs --port=6007，就可以生成网页连接看生成的数据图片了
image_path = "C:\\Users\\HP\\Desktop\\数据集\\hymenoptera_data\\train\\bees_image\\1508176360_2972117c9d.jpg"
img_pil = Image.open(image_path)
img_array= np.array(img_pil)   #转换成array类型


writer.add_image("test", img_array, 2, dataformats='HWC')  #image的shape为高宽通道
# y = x
for i in range(100):
    writer.add_scalar("y=2x", 2*i, i)  #tag：图片标题，后面i和i分别未x轴和Y轴

writer.close()