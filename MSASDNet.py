import torch
import time
import math
import torchvision
import torch.nn as nn
from backbones.MASSDNet_backbone import resnet_MASSD, ResNet


# 卷积>归一化>激活函数
def CBR(in_channel, out_channel, kernel_size, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True)
    )


def upsample(in_channel, out_channel):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True)
    )


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class MSASDNet(nn.Module):
    def __init__(self, backbone=resnet_MASSD):
        super(MSASDNet, self).__init__()

        self.backbone = backbone()
        self.pool1 = nn.MaxPool2d(8, 8)
        self.pool2 = nn.MaxPool2d(4, 4)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.pool4 = nn.MaxPool2d(1, 1)

        # 1x1卷积减少通道数512 > 64
        self.conv1_1 = CBR(512, 64, 1)
        self.conv2_1 = CBR(512, 64, 1)
        self.conv3_1 = CBR(512, 64, 1)
        self.conv4_1 = CBR(512, 64, 1)

        # 分支反卷积上采样 kernel_size=4, stride=2, padding=1 可实现2倍上采样
        self.up1_1 = upsample(64, 64)
        self.up1_2 = upsample(64, 64)
        self.up1_3 = upsample(64, 64)

        self.up2_1 = upsample(64, 64)
        self.up2_2 = upsample(64, 64)

        self.up3_1 = upsample(64, 64)

        # 空间注意力
        self.sa = SpatialAttention()
        self.sa1 = SpatialAttention()
        self.sa2 = SpatialAttention()
        self.sa3 = SpatialAttention()
        self.sa4 = SpatialAttention()

        # 1x1卷积减少通道数512 > 64
        self.conv_main = CBR(512, 64, 1)

        # 特征组合后反卷积上采样
        self.conv1 = upsample(256, 64)
        self.conv2 = upsample(64, 32)

        # 预测部分
        self.conv3 = CBR(32, 2, 1)

        # 是否为阴影概率预测 参数是维度dim，表示在那个维度进行softmax
        self.softmax = nn.Softmax(1)


    def forward(self, x):
        out = self.backbone(x)
        out1 = self.pool1(out)
        out2 = self.pool2(out)
        out3 = self.pool3(out)
        out4 = self.pool4(out)

        out1 = self.conv1_1(out1)
        out2 = self.conv2_1(out2)
        out3 = self.conv3_1(out3)
        out4 = self.conv4_1(out4)

        out1 = self.up1_1(out1)
        out1 = self.up1_2(out1)
        out1 = self.up1_3(out1)

        out2 = self.up2_1(out2)
        out2 = self.up2_2(out2)

        out3 = self.up3_1(out3)

        # 主分支1x1卷积减少通道数
        # out = self.conv_main(out)

        # 添加空间注意力
        # print(self.sa1(out1).size())
        # out = self.sa(out) * out
        out1 = self.sa1(out1) * out1
        out2 = self.sa1(out2) * out2
        out3 = self.sa1(out3) * out3
        out4 = self.sa1(out4) * out4


        # 特征组合
        out = torch.cat((out1, out2, out3, out4), 1)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.softmax(out)

        # print(out.size())
        # print(out1.size())
        # print(out2.size())
        # print(out3.size())
        # print(out4.size())

        return out


if __name__ == '__main__':
    begin = time.time()
    resnet = MSASDNet(resnet_MASSD)
    x = torch.rand(2, 3, 256, 256)
    y = resnet(x)
    #print(resnet)
    print(y[0,0,0,1], y[0,1,0,1])

    end = time.time()
    print("花费时间：", end-begin, "s")


