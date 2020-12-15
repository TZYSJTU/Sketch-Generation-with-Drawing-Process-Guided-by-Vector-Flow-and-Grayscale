import cv2
import numpy as np
import math
import torch
import torch.nn as nn 
import torchvision.transforms as transforms
import torch.nn.functional as F 
from PIL import Image , ImageFilter
import matplotlib.pyplot as plt
import time 
import sys
import os

from LDR import *
from tone import *
from genStroke_origin import *

from drawpatch import rotate
from tools import *
from ETF.edge_tangent_flow import *
from Edge.pencil import *
from deblue import deblue
from extract import extract_pro

# args
input_path = './input/jm.png'
output_path = './output' 

np.random.seed(1)
n =  7                 # Quantization order
period = 4              # line period
direction =  10         # num of dir
Freq = 100              # save every（freq) lines drawn
deepen =  1             # for edge
transTone = False       # for Tone
kernel_radius = 3       # for ETF
iter_time = 15          # for ETF
background_dir = None   # for ETF 
CLAHE = True
edge_CLAHE = True
draw_new = True



if __name__ == '__main__': 
    ####### ETF #######
    time_start=time.time()
    ETF_filter = ETF(input_path=input_path, output_path=output_path+'/mask',\
         dir_num=direction, kernel_radius=kernel_radius, iter_time=iter_time, background_dir=background_dir)
    ETF_filter.forward()
    print('ETF done')


    input_img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    (h0,w0) = input_img.shape
    cv2.imwrite(output_path + "/input_gray.png", input_img)
    # if h0>w0:
    #     input_img = cv2.resize(input_img,(int(256*w0/h0),256))
    # else:
    #     input_img = cv2.resize(input_img,(256,int(256*h0/w0)))    
    # (h0,w0) = input_img.shape

    if transTone == True:
        input_img = transferTone(input_img)
    
    now_ = np.uint8(np.ones((h0,w0)))*255
    step = 0
    if draw_new==True:
        time_start=time.time()
        for dirs in range(direction):
            
            angle = -90+dirs*180/direction
            img,_ = rotate(input_img, -angle)

            ############ Adjust Histogram ############
            if CLAHE==True:
                img = HistogramEqualization(img)
            # cv2.imshow('HistogramEqualization', res)
            # cv2.waitKey(0)
            # cv2.imwrite(output_path + "/HistogramEqualization.png", res)
            print('HistogramEqualization done')

            ############   Quantization   ############ 
            ldr = LDR(img, n)
            # cv2.imshow('Quantization', ldr)
            # cv2.waitKey(0)
            cv2.imwrite(output_path + "/Quantization.png", ldr)

            # LDR_single(ldr,n,output_path) # debug
            ############     Cumulate     ############
            LDR_single_add(ldr,n,output_path)
            print('Quantization done')
            

            # get tone
            (h,w) = ldr.shape
            canvas = Gassian((h+4*period,w+4*period), mean=250, var = 3)


            for j in range(n):
                print('tone:',j)
                distribution = ChooseDistribution(period=period,Grayscale=j*256/n)
                mask = cv2.imread(output_path + '/mask/mask{}.png'.format(j),cv2.IMREAD_GRAYSCALE)/255
                dir_mask = cv2.imread(output_path + '/mask/dir_mask{}.png'.format(dirs),cv2.IMREAD_GRAYSCALE)
                # if angle==0:
                #     dir_mask[::] = 255
                dir_mask,_ = rotate(dir_mask, -angle, pad_color=0)
                dir_mask[dir_mask<128]=0
                dir_mask[dir_mask>127]=1

                distensce = Gassian((1,int(h/period)+4), mean = period, var = 1)
                distensce = np.uint8(np.round(np.clip(distensce, period*0.8, period*1.25)))
                raw = -int(period/2)

                for i in np.squeeze(distensce).tolist():
                    if raw < h:    
                        y = raw + 2*period
                        raw += i        
                        for interval in get_start_end(mask[y-2*period]*dir_mask[y-2*period]):

                            begin = interval[0]
                            end = interval[1]

                            length = end - begin
                            
                            begin -= 2*period
                            end += 2*period
                            length = end - begin

                            newline = Getline(distribution=distribution, length=length)
                            if length<1000 or begin == -2*period or end == w-1+2*period:
                                temp = canvas[y-int(period/2):y-int(period/2)+2*period,2*period+begin:2*period+end]
                                m = np.minimum(temp, newline[:,:temp.shape[1]])
                                canvas[y-int(period/2):y-int(period/2)+2*period,2*period+begin:2*period+end] = m
                            else:
                                temp = canvas[y-int(period/2):y-int(period/2)+2*period,2*period+begin-2*period:2*period+end+2*period]
                                m = np.minimum(temp, newline)
                                canvas[y-int(period/2):y-int(period/2)+2*period,2*period+begin-2*period:2*period+end+2*period] = m
                            
                        
                            if step % Freq == 0:
                                if step > Freq: # not first time 
                                    before = cv2.imread(output_path + "/process/{0:04d}.png".format(int(step/Freq)-1), cv2.IMREAD_GRAYSCALE)
                                    now,_ = rotate(canvas[2*period:2*period+h,2*period:2*period+w], angle)
                                    (H,W) = now.shape
                                    now = now[int((H-h0)/2):int((H-h0)/2)+h0, int((W-w0)/2):int((W-w0)/2)+w0]
                                    now = np.minimum(before,now)
                                else: # first time to save
                                    now,_ = rotate(canvas[2*period:2*period+h,2*period:2*period+w], angle)
                                    (H,W) = now.shape
                                    now = now[int((H-h0)/2):int((H-h0)/2)+h0, int((W-w0)/2):int((W-w0)/2)+w0]
                                
                                cv2.imwrite(output_path + "/process/{0:04d}.png".format(int(step/Freq)), now)
                                # cv2.imshow('step', canvas)
                                # cv2.waitKey(0)

                            now,_ = rotate(canvas[2*period:2*period+h,2*period:2*period+w], angle)
                            (H,W) = now.shape
                            now = now[int((H-h0)/2):int((H-h0)/2)+h0, int((W-w0)/2):int((W-w0)/2)+w0]       
                            now = np.minimum(now,now_)                   
                            step += 1
                            cv2.imshow('step', now_)
                            cv2.waitKey(1)       
                            now_ = now      

                now,_ = rotate(canvas[2*period:2*period+h,2*period:2*period+w], angle)
                (H,W) = now.shape
                now = now[int((H-h0)/2):int((H-h0)/2)+h0, int((W-w0)/2):int((W-w0)/2)+w0]                          
                cv2.imwrite(output_path + "/pro/{}_{}.png".format(dirs,j), now)            

            now,_ = rotate(canvas[2*period:2*period+h,2*period:2*period+w], angle)
            (H,W) = now.shape
            now = now[int((H-h0)/2):int((H-h0)/2)+h0, int((W-w0)/2):int((W-w0)/2)+w0]
            cv2.imwrite(output_path + "/{:.1f}.png".format(angle), now)
            cv2.destroyAllWindows()

        time_end=time.time()
        print('total time',time_end-time_start)
        print('stoke number',step)
        cv2.imwrite(output_path + "/draw.png", now_)
        cv2.imshow('draw', now_)
        cv2.waitKey(0) 
        


    ############ gen edge ###########

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # pc = PencilDraw(device=device, gammaS=1)
    # pc(input_path)
    # edge = cv2.imread('output/Edge.png', cv2.IMREAD_GRAYSCALE)

    edge = genStroke(input_img,18)
    edge = np.power(edge, deepen)
    edge = np.uint8(edge*255)
    if edge_CLAHE==True:
        edge = HistogramEqualization(edge)

    cv2.imwrite(output_path + '/edge.png', edge)
    cv2.imshow("edge",edge)
    cv2.waitKey(0)

    ############# merge #############
    edge = np.float32(edge)
    now_ = cv2.imread(output_path + "/draw.png", cv2.IMREAD_GRAYSCALE)
    result = res_cross= np.float32(now_)

    result[1:,1:] = np.uint8(edge[:-1,:-1] * res_cross[1:,1:]/255)
    result[0] = np.uint8(edge[0] * res_cross[0]/255)
    result[:,0] = np.uint8(edge[:,0] * res_cross[:,0]/255)
    result = edge*res_cross/255
    result=np.uint8(result)  

    cv2.imwrite(output_path + '/result.png', result)
    cv2.imshow("result",result)
    cv2.waitKey(0)
    
    # deblue
    deblue(result, output_path)

    # RGB
    img_rgb_original = cv2.imread(input_path, cv2.IMREAD_COLOR)
    cv2.imwrite(output_path + "/input.png", img_rgb_original)
    img_yuv = cv2.cvtColor(img_rgb_original, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = result
    img_rgb = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR) 

    cv2.imshow("RGB",img_rgb)
    cv2.waitKey(0)
    cv2.imwrite(output_path + "/result_RGB.png",img_rgb)

    # extract drawing process
    extract_pro(input_path=input_path, output_path=output_path+'/step', Quantization=n, direction=direction)