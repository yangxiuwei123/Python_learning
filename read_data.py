from torch.utils.data import Dataset
from PIL import Image
import os

#pytorch加载数据
"""
Dataset:提供一种方式去获取数据及其对应的真实label值
    1、如何获取每一个数据及其label
    2、总共有多少的数据
    数据集文件下有ants和bees两个label文件
Dataloader:为后面的网络提供不同的数据形式（打包）
"""

#1、Dataset
class MyData(Dataset): #继承Dataset类
    def __init__(self, root_dir, label_dir): #初始化,为整个class提供全局变量
        self.root_dir = root_dir   #C:\\Users\\HP\\Desktop\\数据集\\hymenoptera_data\\train
        self.label_dir = label_dir #\\ants
        self.path = os.path.join(self.root_dir + self.label_dir) #C:\\Users\\HP\\Desktop\\数据集\\hymenoptera_data\\train\\ants
        self.image_path = os.listdir(self.path) #每一个图片的名字组成的列表"idx"

    def __getitem__(self, idx): #idx：编号
        img_name = self.image_path[idx] #获取每一张图片的名字
        img_item_path = os.path.join(self.root_dir , self.label_dir, img_name) #获取每一个图片的地址
        img = Image.open(img_item_path)
        label = self.label_dir

        return img, label

    def __len__(self):
        return len(self.image_path)

root_dir = "C:\\Users\\HP\\Desktop\\数据集\\hymenoptera_data\\train\\"
ants_label_dir = "ants"
bees_label_dir = "bees"
ants_dataset = MyData(root_dir, ants_label_dir)
bees_dataset = MyData(root_dir, bees_label_dir)

train_dataset = ants_dataset + bees_dataset
img,label = train_dataset[200]
img.show()
# ants_img,ants_label = ants_dataset[0]
# bees_img,bees_label = bees_dataset[0]
# ants_img.show()
# bees_img.show()


