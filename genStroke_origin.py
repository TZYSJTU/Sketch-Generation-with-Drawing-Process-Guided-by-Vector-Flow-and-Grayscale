import cv2
import numpy as np
from matplotlib import pyplot as plt

import math
import sys
import os

# compute the kernal of different direction
def rotateImg(img, angle):
    row, col = img.shape
    M   = cv2.getRotationMatrix2D((row / 2 , col / 2 ), angle, 1)
    res = cv2.warpAffine(img, M, (row, col))
    return res      
    

# compute and get the stroke of the raw img
def genStroke(img, dirNum, verbose = False):
    height , width = img.shape[0], img.shape[1]
    img = np.float32(img) / 255.0
    print("Input  height: %d, width: %d"%(height,width)) 

    print("PreProcessing Images, denoising ...") 
    img = cv2.medianBlur(img, 3)
    # if verbose == True:
    #     cv2.imshow('blurred image', np.uint8(img*255))
    #     cv2.waitKey(0)

    print("Generating Gradient Images ...")
    imX = np.append(np.absolute(img[:, 0 : width - 1]  - img[:, 1 : width]),  np.zeros((height, 1)), axis = 1)
    imY = np.append(np.absolute(img[0 : height - 1, :] - img[1 : height, :]), np.zeros((1, width)), axis = 0)
##############################################################
#####   Here we have many methods to generate gradient   #####
##############################################################
    img_gradient = np.sqrt((imX ** 2 + imY ** 2))
    img_gradient = imX + imY
    if verbose == True:
        cv2.imshow('gradient image', np.uint8(255-img_gradient*255))
        cv2.imwrite('output/grad.jpg',np.uint8(255-img_gradient*255))
        cv2.waitKey(0)



    #filter kernel size
    tempsize = 0 
    if height > width:
        tempsize = width
    else:
        tempsize = height
    tempsize /= 30
#####################################################################
# according to the paper, the kernelsize is 1/30 of the side length 
#####################################################################
    halfKsize = int(tempsize / 2)
    if halfKsize < 1:
        halfKsize = 1
    if halfKsize > 9:
        halfKsize = 9
    kernalsize = halfKsize * 2 + 1
    print("Kernel Size = %s" %(kernalsize)) 



##############################################################
############### Here we generate the kernal ##################
##############################################################
    kernel = np.zeros((dirNum, kernalsize, kernalsize))
    kernel [0,halfKsize,:] = 1.0
    for i in range(0,dirNum):
        kernel[i,:,:] = temp = rotateImg(kernel[0,:,:], i * 180 / dirNum)
        kernel[i,:,:] *= kernalsize/np.sum(kernel[i])
        # print(np.sum(kernel[i]))
        if verbose == True:
            # print(kernel[i])
            title = 'line kernel %d'%i
            cv2.imshow( title, np.uint8(temp*255))
            cv2.waitKey(0)

#####################################################
# cv2.filter2D() 其实做的是correlate而不是conv
# correlate 相当于 kernal 旋转180° 的 conv
# 但是我们的kernal是中心对称的，所以不影响 
#####################################################

    #filter gradient map in different directions
    print("Filtering Gradient Images in different directions ...") 
    response = np.zeros((dirNum, height, width))
    for i in range(dirNum):
        ker = kernel[i,:,:]; 
        response[i, :, :] = cv2.filter2D(img_gradient, -1, ker)
    if verbose == True:
        for i in range(dirNum):
            title = 'response %d'%i
            cv2.imshow(title, np.uint8(response[i,:,:]*255))
            cv2.waitKey(0)



    #divide gradient map into different sub-map
    print("Caculating Gradient classification ...")
    Cs = np.zeros((dirNum, height, width))
    for x in range(width):
        for y in range(height):
            i = np.argmax(response[:,y,x])
            Cs[i, y, x] = img_gradient[y,x]
    if verbose == True:
        for i in range(dirNum):
            title = 'max_response %d'%i
            cv2.imshow(title, np.uint8(Cs[i,:,:]*255))
            cv2.waitKey(0)



    #generate line shape
    print("Generating shape Lines ...")
    spn = np.zeros((dirNum, height, width))
    for i in range(dirNum):
        ker = kernel[i,:,:]; 
        spn[i, :, :] = cv2.filter2D(Cs[i], -1, ker)
    sp = np.sum(spn, axis = 0)

    sp = sp * np.power(img_gradient, 0.4) 
    ################# 这里怎么理解看论文 #################
    sp =  (sp - np.min(sp)) / (np.max(sp) - np.min(sp))
    S  = 1 -  sp
    # if verbose == True:
    #     cv2.imshow('raw stroke', np.uint8(S*255))
    #     cv2.waitKey(0)

    return S

if __name__ == '__main__':

    img_path   = './input/1.jpg'
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    stroke = genStroke(img, 18, False)
    #stroke = stroke*(np.exp(stroke)-np.exp(1)+1)
    stroke=np.power(stroke, 3)
    # stroke=(stroke - np.min(stroke)) / (np.max(stroke) - np.min(stroke)) # Deepen the edges
    stroke = np.uint8(stroke*255)

    cv2.imwrite('output/edge.jpg',stroke)
    cv2.imshow('stroke', stroke)
    cv2.waitKey(0)

