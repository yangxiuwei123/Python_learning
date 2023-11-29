import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset_transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

#CIFAR10数据集 root要存放的路径，train是否为训练集， download是否下载
train_set = torchvision.datasets.CIFAR10(root="./dataset",train=True, transform=dataset_transforms, download=True)
test_set = torchvision.datasets.CIFAR10(root="./dataset",train=False, transform=dataset_transforms, download=True)

# print(test_set.classes)#所有类别['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
# img, target = test_set[0] #target为该图片对应的类别
# img.show()
writer = SummaryWriter("dataset_transforms")
for i in range(10):
    img,target = test_set[i]
    writer.add_image("test_set",img,i)

writer.close()