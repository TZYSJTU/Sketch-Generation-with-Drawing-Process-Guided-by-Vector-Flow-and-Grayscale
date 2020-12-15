import cv2
import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from simulate import *
from drawpatch import *


def LDR(img, n):
    Interval = 255.0/n
    img = np.float32(img)
    img = np.uint8(img/Interval)
    img = np.clip(img,0,n-1)
    img = np.uint8((img+0.5)*Interval)

    return img


def HistogramEqualization(img,clipLimit=2, tileGridSize=(10,10)):
    # create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    img = clahe.apply(img)
    return img

if __name__ == '__main__':
    img_path   = './input/jiangwen/010s.jpg'
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    img = HistogramEqualization(img)
    # img = LDR(img)
    # LDR_single(img, 8)

    # # 核的大小和形状
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    # # 开操作
    # img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=3)
    # cv2.imshow('MORPH_OPEN', img)
    # cv2.waitKey(0)

    # LDR_single_add(img, 8)

    s = SideWindowFilter(radius=1, iteration=1)
    img = torch.tensor(img, dtype=torch.float32)

    if len(img.size()) == 2:
        h, w = img.size()
        img = img.view(-1, 1, h, w)
    else:
        c, h, w = img.size()
        img = img.view(-1, c, h, w)
    print('img size ', img.shape)

    res = s.forward(img)
    print('res = ', res.shape)
    if res.size(1) == 2:
        res = np.transpose(np.squeeze(res.data.numpy()), (1, 2, 0))
    else:
        res = np.squeeze(res.data.numpy())


    for n in range(8,11):
        img = LDR(res, n)
        # cv2.imshow("LDR",img)
        # cv2.waitKey(0)
        cv2.imwrite("D:/ECCV2020/input/jiangwen/LDR{}.jpg".format(n),img)

    print("done")
    

