import torch
import cv2
import torch.nn as nn
from MSASDNet import MSASDNet
from make_dataset import transform
import numpy as np
import sys
import os
import argparse

# 计算各指标
def Evaluator(img_out, gt):

    # 计算指标 img_out和gt 进行比较
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for i in range(img_out.shape[0]):
        for j in range(img_out.shape[1]):

            if img_out[i, j] == 255 and gt[i, j] == 255:
                TP += 1

            elif img_out[i, j] == 255 and gt[i, j] == 0:
                FP += 1

            elif img_out[i, j] == 0 and gt[i, j] == 255:
                FN += 1


            elif img_out[i, j] == 0 and gt[i, j] == 0:
                TN += 1


    accuracy = (TP+TN) * 100/(TP+TN+FP+FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F_Score = TP * 2 / (2*TP + FP + FN)
    BER = (1 - 0.5 * (TP / (TP + FN) + TN / (TN + FP)))

    print("accuracy  = ", (TP+TN) * 100/(TP+TN+FP+FN), "%")
    print("precision = ", TP * 100 / (TP + FP), "%")
    print("recall    = ", TP * 100 / (TP + FN), "%")
    print("F-Score   = ", TP * 200 / (2*TP + FP + FN), "%")
    print("BER       = ", BER * 100, "%")


    return accuracy, precision, recall, F_Score



# 输入一张图片输出阴影检测结果显示

def predict(img_ori, model_path, gt=None):
    """
    :param img_ori: 原图
    :param gt: 真值图
    :param model_path:模型路径
    :return: None
    """

    # 加载模型
    model = MSASDNet().cuda()
    with torch.no_grad():
        model.load_state_dict(torch.load(model_path))
        img_ori = cv2.imread(img_ori)
        img_ori = cv2.resize(img_ori, dsize=(512, 512))


        cv2.imshow("img_ori", img_ori)

        img = transform(img_ori)
        imgs = torch.zeros(1, 3, img.shape[1], img.shape[2])
        imgs[0] = img
        # print(imgs.shape)

        img_out = model(imgs.cuda())
        # print(img_out.shape)

        img_out = img_out.cpu().data.numpy()
        # print(img_out.shape)
        img_out = img_out[0, 1, :, :]
        # print(np.max(img_out))
        # > 0.5 == 255  <= 0.5== 0 数据类型装成uint8
        img_out[img_out > 0.5] = 255
        img_out[img_out <= 0.5] = 0
        img_out.dtype == "uint8"


        cv2.imshow("result", img_out)


        if gt is not None:
            gt = cv2.imread(gt, 0)
            gt = cv2.resize(gt, dsize=(512, 512))
            cv2.imshow("gt", gt)
            accuracy, precision, recall, F_Score = Evaluator(img_out, gt)


    cv2.waitKey(0)



# 测试单个图像并显示
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', default='./austin22_sub4.tif', type=str, help='The path of shadow image!')
    parser.add_argument('--gt', default=None, type=str, help='The path of ground truth!')

    args = parser.parse_args()
    predict(img_ori=args.img, gt=args.gt, model_path='./models/model.pth')







