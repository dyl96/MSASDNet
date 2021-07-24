import torch
import math
import torchvision
import torch.nn as nn

BatchNorm = nn.BatchNorm2d

# 3x3卷积
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

# ResNet Basicblock
class Basicblock(nn.Module):
    expansion = 1    # 残差块通道数变化倍数

    # in_channel :第一层的输入
    # out_channel:第一层的输出
    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(Basicblock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out



# 定义Bottleneck
# downsample:特征图是否下采样
class Bottleneck(nn.Module):
    expansion = 4   # 经过一次该模块特征图通道数增加四倍

    # in_channel :第一层的输入
    # out_channel:第一层的输出
    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)

        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

        self.conv3 = nn.Conv2d(out_channel, out_channel * 4, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


# 定义一个ResNet骨干网络
class ResNet(nn.Module):
    def __init__(self, block, blocks_num, num_classes=1000, include_top=True):
        self.in_channel = 64    # 残差层的输入channel
        self.include_top = include_top
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 定义残差层
        self.layer1 = self._make_layer(block, 64,  blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=1)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=1)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)

        if self.include_top is True:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output_size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 初始化权重
        self._init_weight()


        # 三个3*3代替7*7
        # self.conv1 = conv3x3(3, 64, stride=2)
        # self.bn1 = nn.BatchNorm2d(64)
        # self.relu1 = nn.ReLU(inplace=True)
        # self.conv2 = conv3x3(64, 64)
        # self.bn2 = nn.BatchNorm2d(64)
        # self.relu2 = nn.ReLU(inplace=True)
        # self.conv3 = conv3x3(64, 128)
        # self.bn3 = nn.BatchNorm2d(128)
        # self.relu3 = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        if self.include_top is True:
            out = self.avgpool(out)
            out = torch.flatten(out, 1)
            out = self.fc(out)

        return out

    # channel:第一层卷积层输出
    # block_num:有几个这个块
    def _make_layer(self, block, channel, block_num, stride=1):
        downsample=None
        # 输入特征图尺寸与输出特征图尺寸不匹配
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel*block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion)
            )

        layers = []
        layers.append(block(self.in_channel, channel, downsample=downsample, stride=stride))
        self.in_channel = channel * block.expansion

        for i in range(1, block_num):
            layers.append(block(self.in_channel, channel))

        return nn.Sequential(*layers)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()





def resnet34(num_classes=1000, include_top=False):
    return ResNet(Basicblock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)

def resnet101(num_classes=1000, include_top=False):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)

def resnet_MASSD(num_classes=1000, include_top=False):
    return ResNet(Basicblock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)



if __name__ == '__main__':
    resnet = resnet_MASSD()
    x = torch.rand(1, 3, 256, 256)
    y = resnet(x)

    print(x.size())
    print(y.size())
