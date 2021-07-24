import cv2
import numpy as np
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import time
from make_dataset import ImgDataset, readfile, readfile_add
from MSASDNet import MSASDNet


f = open("./logs/train_logs.txt", 'w')
f.write("Training loss logs:")
f.write("\n")
f.close()

#
train_path = "./AISD/Train412/"
val_path = "./AISD/Val51/"
print("Reading data ......")
batch_size = 4
train_x, train_y = readfile_add(train_path, True)
val_x, val_y = readfile(val_path, True)
train_set = ImgDataset(train_x, train_y)
val_set = ImgDataset(val_x, val_y)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)


# 定义模型
model = MSASDNet().cuda()
loss = nn.CrossEntropyLoss()
# 二分类交叉熵损失函数
loss = nn.CrossEntropyLoss()
# 优化器 Adam
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epoch = 30

# 开始训练
time_start = time.time()

for epoch in range(num_epoch):
    f = open("./logs/train_logs.txt", 'a')

    epoch_start_time = time.time()
    train_f_score = 0.0
    train_loss = 0.0
    val_f_score = 0.0
    val_loss = 0.0
    model.train()  # 开启BatchNormlization 和 Dropout
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()  # grad归0
        train_pred = model(data[0].cuda())
        # print(train_pred.size())
        # print(data[1].size())
        batch_loss = loss(train_pred, data[1].cuda())
        batch_loss.backward()
        optimizer.step()

        with torch.no_grad():
            # 计算loss和F-score
            train_loss += batch_loss.item()
            # _f_score = f_score(train_pred, data[1].cuda())
            # train_f_score += _f_score

    # 验证集验证
    model.eval()  # 关闭BatchNormlization 和 Dropout
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            val_pred = model(data[0].cuda())
            batch_loss = loss(val_pred, data[1].cuda())


            val_loss += batch_loss.item()
            # _f_score = f_score(val_pred, data[1].cuda())
            # val_f_score += _f_score

        print('[%03d/%03d] %2.2f sec(s) Train  Loss: %3.6f | Val  Loss: %3.6f' %
              (epoch + 1, num_epoch, time.time() - epoch_start_time,
               train_loss,  val_loss))


        f.write('[%03d/%03d] %2.2f sec(s) Train  Loss: %3.6f | Val  Loss: %3.6f' %
                (epoch + 1, num_epoch, time.time() - epoch_start_time, train_loss,  val_loss))
        f.write("\n")
        f.close()

torch.save(model.state_dict(), "./models/model.pth")

time_end = time.time()
print("训练时间:", time_end - time_start, 's')
f.close()
