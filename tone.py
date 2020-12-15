import cv2
import numpy as np
import math
from LDR import LDR

def transferTone(img):
    ho = np.zeros( 256 )
    po = np.zeros( 256 )
    for i in range(256 ):
        po[i] = np.sum(img == i)
    po = po / np.sum(po)

    #caculate original cumulative histogram
    ho[0] = po[0]
    for i in range(1,256):
        ho[i] = ho[i - 1] + po[i]

    #use parameter from paper.
    omiga1 = 76
    omiga2 = 22
    omiga3 = 2
    p1 = lambda x : (1 / 9.0) * np.exp(-(255 - x) / 9.0)
    p2 = lambda x : (1.0 / (225 - 105)) * (x >= 105 and x <= 225)
    p3 = lambda x : (1.0 / np.sqrt(2 * math.pi *11) ) * np.exp(-((x - 90) ** 2) / float((2 * (11 **2))))
    p  = lambda x : (omiga1 * p1(x) + omiga2 * p2(x) + omiga3 * p3(x)) * 0.01

    prob = np.zeros(256)
    total = 0
    for i in range(256):
        prob[i] = p(i)
        total = total + prob[i]
    prob = prob / total 

    #caculate new cumulative histogram
    histo = np.zeros(256)
    histo[0] = prob[0]
    for i in range(1, 256):
        histo[i] = histo[i - 1] + prob[i]

    Iadjusted = np.zeros((img.shape[0], img.shape[1]))
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            histogram_value = ho[img[x,y]]
            i = np.argmin(np.absolute(histo - histogram_value))
            Iadjusted[x, y] = i

    Iadjusted = np.uint8(Iadjusted) 
    
    cv2.imshow('adjust tone', Iadjusted)
    cv2.waitKey(0)
    J = Iadjusted
    J = cv2.blur(Iadjusted, (3, 3))
    cv2.imshow('blurred adjust tone', J)
    cv2.waitKey(1)
    return J



def LDR_single(img,n,output_path):
    Interval = 250.0/n
    img = np.float32(img)
    img = np.uint8(img/Interval)
    img = np.clip(img,0,n-1)

    for i in range (n):
        mask = (img-i == 0)
        tone = np.uint8(i*Interval*mask + (1-mask)*255)
        cv2.imwrite(output_path + "/tone{}.png".format(i),tone)

    # cv2.imwrite("D:/ECCV2020/input/lilianjie/eeee.png",eeee)
    return 

def LDR_single_add(img,n,output_path):
    Interval = 250.0/n
    img = np.float32(img)
    img = np.uint8(img/Interval)
    img = np.clip(img,0,n-1)
    # img = np.float32(img)
    # eeee = img*0
    mask_add = img*0
    for i in range (n):
        mask = (img-i == 0)
        mask_add += mask
        cv2.imwrite(output_path +"/mask/mask{}.png".format(i),np.uint8(mask_add*255))
        tone = np.uint8((i+0.5)*Interval*mask_add + (1-mask_add)*255)
        # cv2.imshow('tone{}'.format(i), tone)
        # cv2.waitKey(0)
        cv2.imwrite(output_path +"/mask/tone_cumulate{}.png".format(i),tone)

    # cv2.imwrite("D:/ECCV2020/input/lilianjie/eeee.png",eeee)
    return 

# def LDR_single_add(img,n1,n2,output_path):
#     Interval = 250.0/n1
#     img = np.float32(img)
#     img = np.uint8(img/Interval)
#     # img = np.clip(img,0,n1-1)
#     # img = np.float32(img)
#     # eeee = img*0

#     for i in range (n1):
#         if i <n2:
#             mask_add = (img-i == 0)
#         else :
#             mask_add = img*0
#             for j in range(n2):
#                 mask = (img-i-j == 0)
#                 mask_add += mask

#         cv2.imwrite(output_path +"/mask/mask{}.png".format(i),np.uint8(mask_add*255))
#         tone = np.uint8((i)*Interval*mask_add + (1-mask_add)*255)
#         # cv2.imshow('tone{}'.format(i), tone)
#         # cv2.waitKey(0)
#         cv2.imwrite(output_path +"/mask/tone_cumulate{}.png".format(i),tone)

#     # cv2.imwrite("D:/ECCV2020/input/lilianjie/eeee.png",eeee)
#     return 

    
if __name__ == '__main__':
    img_path   = './input/jiangwen/MORPH_OPEN.png'
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # img = transferTone(img)
    # cv2.imwrite("./input/jiangwen/transferTone.png",img)

    LDR_single(img,10)
    LDR_single_add(img,10)
    print("done")