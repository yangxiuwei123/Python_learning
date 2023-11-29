import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import *

#准备数据集
train_data = torchvision.datasets.CIFAR10(root="../dataset", train=True, transform=torchvision.transforms.ToTensor(), download=True)
test_data = torchvision.datasets.CIFAR10(root="../dataset", train=False, transform=torchvision.transforms.ToTensor(), download=True)

train_data_size = len(train_data)  #50000
test_data_size = len(test_data)    #10000

print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))

#利用dataloader来加载数据集
train_dataloader = DataLoader(train_data, 64)
test_dataloader = DataLoader(test_data, 64)

#搭建神经网络  在model文件中
sanqi = Sanqi()

#损失函数
loss_fn = nn.CrossEntropyLoss()

#优化器
learing_rate = 0.01
optimizer = torch.optim.SGD(sanqi.parameters(), lr=learing_rate)

#设置训练网络的一些参数
total_train_step = 0 #记录训练的次数
total_test_step = 0 #记录测试的次数
epoch = 10  #训练的轮数

#添加tensboard
writer = SummaryWriter("./logs_train")

for i in range(epoch):
    print("------第 {} 轮训练开始------".format(i+1))

    #训练步骤开始
    sanqi.train()   #网络设置为训练模式，只对dropout层，batchnorm层等有作用
    for data in train_dataloader:
        imgs, targets = data
        outputs = sanqi(imgs)
        loss = loss_fn(outputs, targets)

        #优化器优化模型
        optimizer.zero_grad()  #梯度清零
        loss.backward()   #反向传播
        optimizer.step()


        total_train_step += 1
        if total_train_step % 100 == 0:
            print("训练次数：{}, Loss：{}".format(total_train_step, loss.item()))  #最好加上item()
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    #测试步骤开始
    sanqi.eval()
    total_test_loss = 0
    total_accuracy = 0  #整体正确个数
    with torch.no_grad():  #测试时不需要对梯度进行调整，将梯度进行清零
        for data in test_dataloader:
            imgs, targets = data
            outputs = sanqi(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
            # 比如二分类问题中一个输出对应outputs输出两个概率值，argmax(1):1为横向看，会返回横向两个概率值最大的那个值的下标0或者1，二分类中target为0或者1，判断两者相等的个数
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy

    print("整体测试集上的Loss：{}".format(total_test_loss))
    print("整体测试集上的正确率：{}".format(total_accuracy / test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy / test_data_size, total_test_step)

    total_test_step += 1

    #保存模型
    torch.save(sanqi, "sanqi_{}.pth".format(i))
    print("模型 sanqi_{}.pth 已保存".format(i))

writer.close()

