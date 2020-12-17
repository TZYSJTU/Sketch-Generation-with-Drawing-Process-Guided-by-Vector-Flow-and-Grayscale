import cv2
import math
import numpy as np
import torch
import torch.nn as nn
# np.set_printoptions(threshold=sys.maxsize)

class ETF():
    def __init__(self, input_path, output_path, dir_num, kernel_radius, iter_time, background_dir=None):

        img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE) 
        self.origin_shape = img.shape
        (h,w) = img.shape
        if h>w:
            img = cv2.resize(img,(int(512*w/h),512))
        else:
            img = cv2.resize(img,(512,int(512*h/w)))
        self.shape = img.shape
        self.kernel_size = kernel_radius*2+1
        self.kernel_radius = kernel_radius
        self.iter_time = iter_time
        self.output_path = output_path
        self.dir_num = dir_num
        self.background_dir = background_dir

        img = cv2.copyMakeBorder(img, kernel_radius, kernel_radius, kernel_radius, kernel_radius, cv2.BORDER_REPLICATE)
        img_normal = cv2.normalize(img.astype("float32"), None, 0.0, 1.0, cv2.NORM_MINMAX)

        x_der = cv2.Sobel(img_normal, cv2.CV_32FC1, 1, 0, ksize=5) 
        y_der = cv2.Sobel(img_normal, cv2.CV_32FC1, 0, 1, ksize=5) 

        x_der = torch.from_numpy(x_der) + 1e-12
        y_der = torch.from_numpy(y_der) + 1e-12

        gradient_magnitude = torch.sqrt(x_der**2.0 + y_der**2.0)
        gradient_norm = gradient_magnitude/gradient_magnitude.max()

        x_norm = x_der/(gradient_magnitude)
        y_norm = y_der/(gradient_magnitude)

        # rotate 90 degrees counter-clockwise
        self.x_norm = -y_norm
        self.y_norm =  x_norm

        self.gradient_norm = gradient_norm
        self.gradient_magnitude = gradient_magnitude

    def Ws(self):
        kernels = torch.ones((*self.shape,self.kernel_size,self.kernel_size))
        # radius = central = (self.kernel_size-1)/2
        # for i in range(self.kernel_size):
        #     for j in range(self.kernel_size):
        #         if (i-central)**2+(i-central)**2 <= radius**2:
        #              self.flow_field[x][y]
        return kernels

    def Wm(self):
        kernels = torch.ones((*self.shape,self.kernel_size,self.kernel_size))

        eta = 1 # Specified in paper
        (h,w) = self.shape
        x = self.gradient_norm[self.kernel_radius:-self.kernel_radius,self.kernel_radius:-self.kernel_radius]
        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                y = self.gradient_norm[i:i+h,j:j+w]
                kernels[:,:,i,j] = (1/2) * (1 + torch.tanh(eta*(y - x)))
        return kernels

    def Wd(self):
        kernels = torch.ones((*self.shape,self.kernel_size,self.kernel_size))

        (h,w) = self.shape
        X_x = self.x_norm[self.kernel_radius:-self.kernel_radius,self.kernel_radius:-self.kernel_radius]
        X_y = self.y_norm[self.kernel_radius:-self.kernel_radius,self.kernel_radius:-self.kernel_radius]
    
        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                Y_x = self.x_norm[i:i+h,j:j+w]
                Y_y = self.y_norm[i:i+h,j:j+w]
                kernels[:,:,i,j] = X_x*Y_x + X_y*Y_y

        return torch.abs(kernels), torch.sign(kernels)

    def forward(self):
        Ws = self.Ws()
        Wm = self.Wm()
        for iter_time in range(self.iter_time):
            Wd, phi = self.Wd()
            kernels = phi*Ws*Wm*Wd 

            x_magnitude = (self.gradient_norm*self.x_norm).unsqueeze(0).unsqueeze(0)
            # print(x_magnitude.min())
            y_magnitude = (self.gradient_norm*self.y_norm).unsqueeze(0).unsqueeze(0)

            x_patch = torch.nn.functional.unfold(x_magnitude, (self.kernel_size, self.kernel_size))
            y_patch = torch.nn.functional.unfold(y_magnitude, (self.kernel_size, self.kernel_size))

            x_patch = x_patch.view(self.kernel_size, self.kernel_size,*self.shape)
            y_patch = y_patch.view(self.kernel_size, self.kernel_size,*self.shape)

            x_patch = x_patch.permute(2,3,0,1)
            y_patch = y_patch.permute(2,3,0,1)
            

            x_result = (x_patch*kernels).sum(-1).sum(-1)
            y_result = (y_patch*kernels).sum(-1).sum(-1)
            
            magnitude = torch.sqrt(x_result**2.0 + y_result**2.0)
            x_norm = x_result/magnitude
            y_norm = y_result/magnitude

            self.x_norm[self.kernel_radius:-self.kernel_radius,self.kernel_radius:-self.kernel_radius] = x_norm
            self.y_norm[self.kernel_radius:-self.kernel_radius,self.kernel_radius:-self.kernel_radius] = y_norm
        
        self.save(x_norm,y_norm)
        return None

    def save(self,x,y):
        x = nn.functional.interpolate(x.unsqueeze(0).unsqueeze(0),[*self.origin_shape],mode='nearest')
        y = nn.functional.interpolate(y.unsqueeze(0).unsqueeze(0),[*self.origin_shape],mode='nearest')
        x = x.squeeze()
        y = y.squeeze()
        x[x==0] += 1e-12

        tan = -y/x
        angle = torch.atan(tan)
        angle = 180*angle/math.pi
        if self.background_dir!=None:
            t = self.gradient_magnitude[self.kernel_radius:-self.kernel_radius,self.kernel_radius:-self.kernel_radius]
            t = nn.functional.interpolate(t.unsqueeze(0).unsqueeze(0),[*self.origin_shape], mode='bilinear')
            t = t.squeeze()
            a = t.min()
            b= t.max()
            angle[t<0.4] = self.background_dir
        
        length = 180/self.dir_num
        for i in range(self.dir_num):
            if i==0:
                minimum = -90
                maximum = -90+length/2
                mask1 = 255*(((angle>minimum)+(angle==minimum))*(angle<maximum))
                maximum = 90
                minimum = 90-length/2
                mask2 = 255*((angle>minimum)+(angle==minimum))
                mask = mask1 + mask2
                cv2.imwrite(self.output_path+'/dir_mask{}.png'.format(i),np.uint8(mask.numpy()))
            else:
                minimum = -90+(i-1/2)*length   
                maximum = minimum + length  
                mask = 255*(((angle>minimum)+(angle==minimum))*(angle<maximum))
                cv2.imwrite(self.output_path+'/dir_mask{}.png'.format(i),np.uint8(mask.numpy()))

        return


# args
input_path = './input/mo.jpg'
output_path = './output/mask'
kernel_radius = 2
direction = 8

if __name__ == '__main__':
    # vector_field = init_field(input_path, 5)
    ETF_filter = ETF(input_path=input_path, output_path=output_path, dir_num=direction, kernel_radius=kernel_radius, iter_time=30)
    ETF_filter.forward()
    print('done')
