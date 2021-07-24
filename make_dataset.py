import cv2
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset


# 读取训练数据并进行数据增强
# path: 数据集路径
# label:是否需要返回label
def readfile_add(path, label):
    # 读取文件夹下的文件
    shadow_dir = sorted(os.listdir(path + "shadow/"))
    # 装数据增强后的原图的标签
    x_list = []
    y_list = []

    # 进行数据增强
    for _, file in enumerate(shadow_dir):
        img = cv2.imread(os.path.join(path, "shadow/", file))
        mask = cv2.imread(os.path.join(path, "mask/", file), 0)
        # print(os.path.join(path, "shadow/", file))
        # print(os.path.join(path, "mask/", file))
        # print(img.shape)
        # print(mask.shape)

        # 数据增强 shape是长宽高， size是元素个数
        if img.shape[0] > 256:
            # print(img.shape[0], "> 256")
            stride = int((img.shape[0] - 256) / 64) + 1
            # data_num += (stride)*(stride)
            # print((stride)*(stride))
            for i in range(stride):
                for j in range(stride):
                    # 左右边界和上下边界
                    x1 = 64 * i
                    x2 = x1 + 256
                    y1 = 64 * j
                    y2 = y1 + 256

                    # 如果超过了边界
                    if x2 >= img.shape[0]:
                        x1 = img.shape[0] - 256
                        x2 = img.shape[0]
                    if y2 >= img.shape[0]:
                        y1 = img.shape[0] - 256
                        y2 = img.shape[0]

                    sub_img = img[x1:x2, y1:y2, :]
                    sub_mask = mask[x1:x2, y1:y2]


                    x_list.append(sub_img)
                    y_list.append(sub_mask)


            # cv2.imshow("all", img)
            # cv2.imshow("sub", sub_img)

            # cv2.waitKey(0)

        if img.shape[0] == 256:
            # print(img.shape[0], "= 256")
            x_list.append(img)
            y_list.append(mask)

    print("训练数据增强后共：", x_list.__len__(), "张图片")
    print("原始数据切片形状：", x_list[0].shape)
    print("标签数据切片形状：", y_list[0].shape)

    if label == True:
        return x_list, y_list

    return x_list


# 读取数据并进行数据增强
# path: 数据集路径
# label:是否需要返回label
def readfile(path, label):
    # 读取文件夹下的文件
    shadow_dir = sorted(os.listdir(path + "shadow/"))
    # 装数据增强后的原图的标签
    x_list = []
    y_list = []

    # 进行数据增强
    for _, file in enumerate(shadow_dir):
        img = cv2.imread(os.path.join(path, "shadow/", file))
        mask = cv2.imread(os.path.join(path, "mask/", file), 0)
        # print(os.path.join(path, "shadow/", file))
        # print(os.path.join(path, "mask/", file))
        # print(img.shape)
        # print(mask.shape)

        # 数据增强 shape是长宽高， size是元素个数
        if img.shape[0] > 256:
            # print(img.shape[0], "> 256")
            stride = int((img.shape[0]-256)/64) + 1
            # data_num += (stride)*(stride)
            # print((stride)*(stride))
            for i in range(stride):
                for j in range(stride):
                    # 左右边界和上下边界
                    x1 = 64 * i
                    x2 = x1 + 256
                    y1 = 64 * j
                    y2 = y1 + 256

                    # 如果超过了边界
                    if x2 >= img.shape[0]:
                        x1 = img.shape[0] - 256
                        x2 = img.shape[0]
                    if y2 >= img.shape[0]:
                        y1 = img.shape[0] - 256
                        y2 = img.shape[0]

                    sub_img = img[x1:x2, y1:y2, :]
                    sub_mask = mask[x1:x2, y1:y2]


                    x_list.append(sub_img)
                    y_list.append(sub_mask)

            # cv2.imshow("all", img)
            # cv2.imshow("sub", sub_img)

            #cv2.waitKey(0)

        if img.shape[0] == 256:
            # print(img.shape[0], "= 256")
            x_list.append(img)
            y_list.append(mask)



    print("验证数据切片后共：", x_list.__len__(), "张图片")
    print("原始数据切片形状：", x_list[0].shape)
    print("标签数据切片形状：", y_list[0].shape)

    # print(x_list[11835].shape)
    # print(y_list[11835].shape)

    # cv2.imshow("z1", x_list[11832])
    # cv2.imshow("z2", x_list[11833])
    # cv2.imshow("z3", x_list[11834])
    # cv2.imshow("z4", x_list[11835])
    # cv2.imshow("z5", y_list[11832])
    # cv2.imshow("z6", y_list[11833])
    # cv2.imshow("z7", y_list[11834])
    # cv2.imshow("z8", y_list[11835])
    # cv2.waitKey(0)

    if label == True:
        return x_list, y_list

    return x_list


# HWC->CHW且进行了归一化
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor()
])


# # 构建数据集：继承Dataset类 重写__len__以及__getitem__方法
class ImgDataset(Dataset):
    def __init__(self, x, y=None):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        X = self.x[index]
        X = transform(X)
        if self.y is not None:
            Y = self.y[index]
            Y = transform(Y)
            Y = torch.LongTensor(Y[0].numpy())

            # Y = torch.LongTensor(Y)

            return X, Y

        return X

if __name__ == '__main__':

    x, y = readfile("./AISD/Train412/", True)
    dataloader = ImgDataset(x, y)
    # print(dataloader.__len__())

    x = x[11835]

    x = transform(x)
    y = y[11835]
    print(y.shape)

    y = transform(y)
    y=y[0]
    # print(x)
    # x = x.transpose(0, 2).transpose(0, 1)
    # x = np.array(x*255, dtype=np.uint8)
    #
    #
    # print(y)
    print(y.shape)
    # cv2.imshow("x", y)
    cv2.waitKey(0)






