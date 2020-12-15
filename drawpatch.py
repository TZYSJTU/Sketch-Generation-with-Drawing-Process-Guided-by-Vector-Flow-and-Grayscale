import cv2
import numpy as np
from matplotlib import pyplot as plt
from simulate import *


def rotate(image, angle, scale=1.0, pad_color=255):
    # grab the dimensions of the image and then determine the center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    
    # grab the rotation matrix (applying the negative of the angle to rotate clockwise)
    # scale can be adjusted
    M = cv2.getRotationMatrix2D(center=(cX, cY), angle=angle, scale=scale)
    # M.shape = 2 * 3
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    #  cos  sin  0
    # -sin  cos  0
    #   0    0   1
    
    # compute the new size of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
 
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # compute the new origin_point of the image 
    origin_point = np.array([nW/2.0-M[0,1]*h/2.0, nH/2.0-M[1,1]*h/2.0]) 
    origin_point = np.round(origin_point)
    origin_point = (int(origin_point[0]), int(origin_point[1]))

    # perform the actual rotation and return the image
    result = cv2.warpAffine(src=image, M=M, dsize=(nW, nH), borderValue=(pad_color,pad_color,pad_color))

    # result[:,origin_point[0]] = 0
    # result[origin_point[1],:] = 0
    return result, origin_point
 

def drawpatch(canvas, patch_size, angle, scale, location, grayscale):
    distribution = ChooseDistribution(period=7,Grayscale=grayscale)
    patch = GetParallel(distribution=distribution, height = patch_size[0], length = patch_size[1], period=distribution.shape[0])

    imgRotation, origin_point = rotate(image=patch, angle=angle, scale=scale)

    (h,w) = imgRotation.shape
    (H,W) = canvas.shape
    pad_canvas = np.zeros((2*h+canvas.shape[0], 2*w+canvas.shape[1]), dtype=np.uint8)
    pad_canvas[h:h+H, w:w+W] = canvas

    Aligned_point = [w+location[0]-origin_point[0], h+location[1]-origin_point[1]]

    m = np.minimum(imgRotation, pad_canvas[Aligned_point[1]:Aligned_point[1]+h, Aligned_point[0]:Aligned_point[0]+w])
    pad_canvas[Aligned_point[1]:Aligned_point[1]+h, Aligned_point[0]:Aligned_point[0]+w] = m  

    return pad_canvas[h:h+H, w:w+W]




if __name__ == '__main__':
    np.random.seed(1945)
    canvas = Gassian((500,400), mean=250, var = 3)
#####################################################################
#######################          args        ########################
#####################################################################
    sequence = (
                [(1000,1000),0 ,1.0,(500,0),208+32], 
                [(1225,1225),75 ,1.0,(-91,341),208+32], 
                [(1450,1450),135 ,1.0,(0,1000),208+32], 
                )
#####################################################################
#####################################################################
#####################################################################
    for j in range(16):
        canvas = Gassian((1000,1000), mean=250, var = 3)
        for i in sequence:
            canvas = drawpatch(canvas=canvas, patch_size=i[0], angle=i[1], scale=i[2], location=i[3], grayscale=j*16)

        # cv2.imshow("drawpatch",canvas)
        # cv2.waitKey(0)
        cv2.imwrite("D:/ECCV2020/simu_patch/{}.jpg".format(j),canvas)

    print("done")


