#Python实现将一张图片放到另一张图片指定的位置上并合成一张图
from PIL import Image
import cv2
import os
import numpy as np
def Pyramid_CreateNewSrc(src, mask, back, posX , posY):
    img_src = src
    img_mask = mask
    img_back = back

    rows, cols, channels = img_mask.shape  # rows，cols最后一定要是前景图片的，后面遍历图片需要用到
    # 遍历替换
    LeftUpLocation = [posX, posY]  # 在新背景图片中的位置
    for i in range(rows):
        for j in range(cols):
            if img_mask[i, j][0] != 0 or img_mask[i, j][1] != 0 or img_mask[i, j][2] != 0:  # 0代表黑色的点
                if LeftUpLocation[0] + i < img_back.shape[0] and LeftUpLocation[1] + j < img_back.shape[1]:
                    img_back[LeftUpLocation[0] + i, LeftUpLocation[1] + j] = img_src[i, j]  # 此处替换颜色，为BGR通道
    #cv2.imshow('res', img_back)
    return img_back