import cv2
import numpy as np
from matplotlib import pyplot as plt
import math

def Gassian(size, mean = 0, var = 0):
    norm = np.random.randn(*size)
    denorm = norm * np.sqrt(var) + mean
    return np.uint8(np.round(np.clip(denorm,0,255)))

def Getline(distribution, length):
    period = distribution.shape[0]
    if length < 100:  # if length is too short, lines are Aligned
        patch = Gassian((2*period, length), mean=250, var = 3) 
        begin = 0
        end = 1
        for i in range(period):
            patch[i]=Gassian((1,length), mean=distribution[i,0], var=distribution[i,1])

    else:           # if length is't too short, lines is't Aligned
        patch = Gassian((2*period, length+4*period), mean=250, var = 3) 

        begin = Gassian((1,1), mean=2.0*period, var=2*period)
        # egin = Gassian((1,1), mean=2.0*period, var=0)
        begin = np.uint8(np.round(np.clip(begin,0,4*period)))
        begin = int(begin[0,0])
        end = Gassian((1,1), mean=2.0*period, var=2*period)
        # end = Gassian((1,1), mean=2.0*period, var=0)
        end = np.uint8(np.round(np.clip(end,1,4*period+1)))
        end = int(end[0,0])

        real_length = length+4*period-end-begin
        for i in range(period):
            patch[i,begin:-end]=Gassian((1,real_length), mean=distribution[i,0], var=distribution[i,1])

    patch = Attenuation(patch, period=period, distribution=distribution,begin=begin, end=end)
    patch = Distortion(patch, begin=begin, end=end)

    return np.uint8(np.round(np.clip(patch,0,255)))

def Attenuation(patch, period, distribution, begin, end):
    order = int((patch.shape[1]-begin-end)/2)+1
    radius = (period-1)/2
    canvas = Gassian((patch.shape[0], patch.shape[1]), mean=250, var=3)
    patch = np.float32(patch)
    canvas = np.float32(canvas)
    for i in range(begin, patch.shape[1]-end+1):
        for j in range(period):
            a = np.abs((1.0-(i-begin)/order)**2)/3
            b = np.abs((1.0-j/radius)**2)*1
            patch[j,i] += (canvas[j,i]-patch[j,i])*np.sqrt(a+b)/1.5
            # patch[j,i] +=  0.75*(canvas[j,i]-patch[j,i]) * (np.abs((1.0-(i-begin)/order)**2))**0.5

    return np.uint8(np.round(np.clip(patch,0,255)))


def Distortion(patch,begin,end):
    height = patch.shape[0]
    length = patch.shape[1]
    patch = np.float32(patch)
    patch_copy = patch.copy()

    central = ((length-begin-end)/2+begin) + np.random.randn()*length/15
    # central = ((length-begin-end)/2+begin)
    if length>100:
        radius = length**2/(2*height)
    else:
        radius = 100**2/(2*height)
    for i in range(length):
        offset = ((central-i)**2)/(2*radius) 
        int_offset = int(offset)
        decimal_offset = offset-int_offset
        for j in range(height):
            if j>int_offset:
                patch[j,i]=int(decimal_offset*patch_copy[j-1-int_offset,i]+(1-decimal_offset)*patch_copy[j-int_offset,i])
            else:
                patch[j,i]= np.random.randn() * np.sqrt(3) + 250
    
    return np.uint8(np.round(np.clip(patch,0,255)))

def GetParallel(distribution, height, length, period):
    if length<100: # constant length
        canvas = Gassian((height+2*period,length), mean=250, var = 3)  
    else: # variable length
        canvas = Gassian((height+2*period,length+4*period), mean=250, var = 3)  

    distensce = Gassian((1,int(height/period)+1), mean = period, var = period/5)
    # distensce = Gassian((1,int(height/period)+1), mean = period, var = 0)
    distensce = np.uint8(np.round(np.clip(distensce, period*0.8,period*1.25)))

    begin = 0
    for i in np.squeeze(distensce).tolist():
        newline = Getline(distribution=distribution, length=length)
        h,w = newline.shape
        # cv2.imshow('line', newline)
        # cv2.waitKey(0)
        # cv2.imwrite("D:/ECCV2020/simu_patch/Line3.jpg",newline)

        if begin < height:
            m = np.minimum(canvas[begin:(begin + h),:], newline)
            canvas[begin:(begin + h),:] = m
            begin += i
        else:
            break

    return canvas[:height,:]

def ChooseDistribution(period, Grayscale):
    distribution = np.zeros((period,2))
    c = period/2.0
    difference = 250-Grayscale
    for i in range(distribution.shape[0]):
        distribution[i][0] = Grayscale + difference*abs(i-c)/c
        distribution[i][1] = np.cos((i-c)/c*(0.5*3.1415929))*difference+difference**2/300

        # distribution[i][0] -= np.cos((i-4)/4.0*(0.5*3.1415929))*difference
        # distribution[i][1] += np.cos((i-4)/4.0*(0.5*3.1415929))*difference

    return distribution 
    

if __name__ == '__main__':
    np.random.seed(1500)
    canvas = Gassian((400,300), mean=250, var = 3)

    # distribution = np.array([[245,31],[238,27],[218,48],[205,33],[214,38],[234,24],[240,42]])
###################################################
###################################################
###################################################
    period = 8
    Grayscale = 160
    H,L = (100,150)
###################################################
###################################################
###################################################
    distribution = ChooseDistribution(period=period, Grayscale=Grayscale)
    print(distribution)
    patch = GetParallel(distribution=distribution, height=H, length=L, period=period)
    (h,w) = patch.shape
    # patch = GetOffsetParallel(offset=4, distribution=distribution, patch_size=(40,200), period_mean=distribution.shape[0], period_var=1)
    # (h,w) = patch.shape
    # canvas[400-int(h/2):400-int(h/2)+h,300-int(w/2):300-int(w/2)+w] = patch
    
    # cv2.imshow('Parallel', patch[:, 2*distribution.shape[0]:w-2*distribution.shape[0]])
    cv2.imshow('Parallel', patch)
    cv2.waitKey(0)
    cv2.imwrite("D:/ECCV2020/simu_patch/Parallel4.jpg",patch)
    print("done")
