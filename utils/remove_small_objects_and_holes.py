import numpy as np
import cv2
from skimage import morphology


def remove_small_objects_and_holes(src, min_size, area_threshold, connectivity=1, in_place=False):

    '''
    :param src: 输入图像uint8
    :param min_size: 小于改尺寸的小物体被去除
    :param area_threshold: 小于该阈值的空洞被填充
    :param connectivity: 邻接模式，1-四邻接，2-8邻接
    :param in_place: False-复制在操作，True-直接操作
    :return: 去除小物体和空洞的图像
    '''

    # cv2.imshow("src", src)
    src[src > 0.5] = True
    src[src <= 0.5] = False
    src.dtype = 'bool'
    # print(src.dtype)
    # print(src)
    src_temp = src

    src1 = morphology.remove_small_objects(src_temp, min_size=min_size, connectivity=connectivity, in_place=in_place)
    src2 = morphology.remove_small_holes(src1, area_threshold=area_threshold, connectivity=connectivity, in_place=in_place)


    # bool 类型转 uint8
    src2.dtype = 'uint8'
    # True转255
    src2[src2 > 0.5] = 255
    src2[src2 <= 0.5] = 0


    # cv2.imshow('result_change', src2)
    # print(src2.dtype)
    # print(src2.shape)
    # print(src2)

    return src2


if __name__ == '__main__':

    src = cv2.imread("./result_chicago22_sub10.png", 0)
    cv2.imshow('src', src)

    mask = cv2.imread("E:\\code\\my_code\\project_py\\MASSDNet\\MASSDNet_Release\\MASSDNet-1\\AISD\\Test51\\mask\\chicago5_sub7.tif")

    print(src.dtype, src.shape)
    img = remove_small_objects_and_holes(src, 64, 5)

    cv2.imshow('mask', mask)

    cv2.imshow('result', img)

    print(img.shape)

    cv2.waitKey(0)
