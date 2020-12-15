import cv2
import numpy as np
from simulate import Gassian

def deblue(img, output_path):
    BGR = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    HSV = cv2.cvtColor(BGR,cv2.COLOR_BGR2HSV)

    size = img.shape

    HSV[:,:,0]= Gassian(size, mean = 15, var = 1)
    HSV[:,:,1]= Gassian(size, mean = 20, var = 2)


    result = cv2.cvtColor(HSV,cv2.COLOR_HSV2BGR)
    result = np.uint8(result)

    cv2.imwrite(output_path+'/aging.jpg', result)
    # cv2.imshow("aging",result)
    # cv2.waitKey(0)

if __name__ == '__main__':
    input_path = './output/draw.png'
    output_path = './output' 
    input_img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    deblue(input_img, output_path)



