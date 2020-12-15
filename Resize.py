import cv2
import numpy as np

input_path = './input/ztfn1.png'
output_path = './output' 

min_length = 320
img = cv2.imread(input_path, cv2.IMREAD_COLOR)
(h,w,c) = img.shape
if h<w:
    img = cv2.resize(img,(int(min_length*w/h),min_length))
else:
    img = cv2.resize(img,(min_length,int(min_length*h/w)))

cv2.imwrite(output_path + "/up.png", img)
cv2.imshow('draw', img)
cv2.waitKey(0) 

