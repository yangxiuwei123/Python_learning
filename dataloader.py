import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

#准备的测试数据集
test_set = torchvision.datasets.CIFAR10(root="./dataset",train=False, transform=torchvision.transforms.ToTensor())

#batch_size每次取四个image及其target打包为images和targets， shuffle是否打乱顺序
test_loader = DataLoader(dataset=test_set, batch_size=64, shuffle=True, num_workers=0, drop_last=True)

#测试数据集中第一张图片及target
img,target = test_set[0]

writer = SummaryWriter("dataloader")
step = 0
for data in test_loader:
    imgs, targets = data
    writer.add_images("test_data_droplist",imgs,step)
    step += 1

writer.close()