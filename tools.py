import cv2
import numpy as np
from matplotlib import pyplot as plt
import math


def get_start_end(mask):
    lines=[]
    Flag = True # no new interval
    for i in range(mask.shape[0]):
        if Flag == True: 
            if mask[i]==1:
                if len(lines)>0 and i-lines[-1][1]<=1: ####### too close
                    Flag = False
                    continue
                else:
                    lines.append([i,i])
                    Flag = False
            else:
                continue
        else:
            if mask[i]==1:
                continue
            else:
                lines[-1][1]=i   
                Flag = True

    if Flag == False:
        lines[-1][1]=i

    return lines

def rotateImg(img, angle):
    row, col = img.shape
    M   = cv2.getRotationMatrix2D((row / 2 , col / 2 ), angle, 1)
    res = cv2.warpAffine(img, M, (row, col))
    return res     

def get_directions(Num_choose, dirNum, img):
    height,width = img.shape
    img = np.float32(img)/255.0
    # print("Input  height: %d, width: %d"%(height,width)) 

    imX = np.append(np.absolute(img[:, 0 : width - 1]  - img[:, 1 : width]),  np.zeros((height, 1)), axis = 1)
    imY = np.append(np.absolute(img[0 : height - 1, :] - img[1 : height, :]), np.zeros((1, width)), axis = 0)

    img_gradient = np.sqrt((imX ** 2 + imY ** 2))
    mask = (img_gradient-0.02)>0
    cv2.imshow('mask',np.uint8(mask*255))
    # img_gradient = imX + imY

    #filter kernel size
    tempsize = 0 
    if height > width:
        tempsize = width
    else:
        tempsize = height
    tempsize /= 30 # according to the paper, the kernelsize is 1/30 of the side length 

    halfKsize = int(tempsize / 2)
    if halfKsize < 1:
        halfKsize = 1
    if halfKsize > 9:
        halfKsize = 9
    kernalsize = halfKsize * 2 + 1
    # print("Kernel Size = %s" %(kernalsize)) 

##############################################################
############### Here we generate the kernal ##################
##############################################################
    kernel = np.zeros((dirNum, kernalsize, kernalsize))
    kernel [0,halfKsize,:] = 1.0
    for i in range(0,dirNum):
        kernel[i,:,:] = temp = rotateImg(kernel[0,:,:], i * 180 / dirNum)
        kernel[i,:,:] *= kernalsize/np.sum(kernel[i])


    #filter gradient map in different directions
    print("Filtering Gradient Images in different directions ...") 
    response = np.zeros((dirNum, height, width))
    for i in range(dirNum):
        ker = kernel[i,:,:]; 
        response[i, :, :] = cv2.filter2D(img_gradient, -1, ker)

    cv2.waitKey(0)

    #divide gradient map into different sub-map
    print("Caculating direction classification ...")
    direction = np.zeros(( height, width))
    for x in range(width):
        for y in range(height):
            direction[y, x] = np.argmax(response[:,y,x])
    #direction = direction*mask

    dirs = np.zeros(dirNum)
    for i in range (dirNum):
        dirs[i]=np.sum((direction-i)==0)
    sort_dirs = np.sort(dirs,axis=0)
    print(dirs,sort_dirs)
    angles = []
    for i in range(Num_choose):
        for j in range (dirNum):
            if sort_dirs[-1-i]==dirs[j]:
                angles.append(j*180/dirNum)
                continue
    return angles

if __name__ == '__main__':
    input_path = './input/lena.jpg'
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    angles = get_directions(4,12,img)