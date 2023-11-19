from torch.utils.tensorboard import SummaryWriter


"""
scalar_value:对应图像Y轴
global_step:对应X轴
"""
#tensorBoard的使用（1）
writer = SummaryWriter("logs")
# logs文件的使用，在终端使用命令tensorboard --logdir=computer_vision\pytorch\logs --port=6007，就可以生成网页连接看生成的数据图片了


# writer.add_image()
# y = x
for i in range(100):
    writer.add_scalar("y=2x", 2*i, i)  #tag：图片标题，后面i和i分别未x轴和Y轴

writer.close()