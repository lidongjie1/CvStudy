import torch
from torch import nn
from torch.utils import data
import MydataSet
import YOLOV1
dataset_dir = r"C:\Users\Adminer\PycharmProjects\ImageProject\data\scripts\VOCtrainval_06-Nov-2007\VOCdevkit\VOC2007\voc2007Yolov1"

train_dataset = MydataSet.MydataSet(dataset_dir,mode="train")
test_dataset = MydataSet.MydataSet(dataset_dir,mode="test")
train_loder = data.DataLoader(dataset=train_dataset,batch_size=8,shuffle=False)
test_loder = data.DataLoader(dataset=test_dataset,batch_size=8,shuffle=False)
data_loder = {"train":test_loder,"test":test_dataset}
device = torch.device("cuda")  # 使用gpu进行训练
net = YOLOV1.YoLoNet()
net = net.to(device)
params = net.parameters()
optimizer = torch.optim.Adam(params,lr=0.01,weight_decay=0)
filename = 'YoLoV1_pth'


def train_mode(net,data_laoder,optimizer,num_epoch):
    net.to(device)
    for epoch in range(num_epoch):
        net.train()
        phase = "train"
        losssum = 0.
        for imgs, labels in data_laoder[phase]:
            labels1 = labels.view(8, 7, 7, -1)
            labels1 = labels1.permute(0, 3, 1, 2)
            imgs = imgs.to(device)
            labels1 = labels1.to(device)
            preds = net(imgs)  # 前向传播
            loss = net.calculate_loss(preds, labels1)  # 计算损失
            optimizer.zero_grad()  # 梯度清零
            loss.backward()  # 反向传播
            optimizer.step()  # 优化网络参数
            losssum += loss.item()

        epoch_loss = losssum / len(data_loder[phase].dataset)
        print('{} Loss: {:.4f}'.format(phase, epoch_loss))

    torch.save(net.state_dict(), filename)


train_mode(net,data_loder,optimizer,10)



