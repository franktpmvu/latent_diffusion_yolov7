import os 
import cv2
import numpy as np
import random
import copy
from scipy.stats import norm
from matplotlib import pyplot as plt
import torch
import math

class mix_augmentaion(object):
    def __init__(self,imshape=(256,256),noiseStepMode='direct'):
        self.imshape=imshape
        self.noiseStepMode=noiseStepMode
        if noiseStepMode=='exp':
            self.multi_base=0.0001
            self.exp_base=self.compute_pow(self.multi_base,1,99)
        self.random_parameter()
        
    def random_parameter(self,h=None,w=None,rain_slant=None,spotlignt_position=None,spotlight_reverse=None,spotlight_transparency=None,gamma=None,pixelate_blocks=None):
        #rain
        if rain_slant is None:
            rain_slant = np.random.randint(-10, 10) # generate random slant if no slant value is given
        #spot_light
        if spotlight_reverse is None:
            spotlight_reverse=False
            if np.random.random(size=1)>0.5:
                spotlight_reverse=True
        if spotlight_transparency is None:
            spotlight_transparency = random.uniform(0.5, 0.85)
        if spotlignt_position is None:
            spotlignt_position = [(random.random(), random.random())]

        #gamma
        if gamma is None:
            gamma = np.random.randint(40,91) / 100.0
        if pixelate_blocks is None:
            
            #print(int(self.imshape[0]*0.2))
            #print(int(self.imshape[0]*0.3))
            pixelate_blocks = np.random.randint(int(self.imshape[0]*0.2),int(self.imshape[0]*0.3))#0.2 : 0.3 hw
        
        self.rain_slant = rain_slant
        self.rain_drops, self.rain_width = self.generate_random_lines(self.imshape, self.rain_slant)


        self.spotlight_reverse = spotlight_reverse
        self.spotlight_transparency = spotlight_transparency
        self.spotlignt_position = spotlignt_position
        self.gamma = gamma
        self.pixelate_blocks = pixelate_blocks
    

        

        
    def generate_random_lines(self,imshape, slant,step_ratio=1.0):
        drops = []
        drop_width = []
        area = imshape[0] * imshape[1]
        drops_num = int(area//np.random.randint(600, 700)*step_ratio)

        for i in range(drops_num): ## If You want heavy rain, try increasing this
            if slant<0:
                x1 = np.random.randint(slant, imshape[1])
            else:
                x1 = np.random.randint(0, imshape[1]-slant)
            drop_length = np.random.randint(10, 60)
            y1 = np.random.randint(0, imshape[0]-drop_length)
            x2 = x1 + slant
            y2 = y1 + drop_length
            drops.append((x1, y1, x2, y2))
            drop_width.append(np.random.randint(1, 3))
            if not len(drops)==len(drop_width):
                print('len(drops)= '+str(len(drops)))
                print('len(drop_width)= '+str(len(drop_width)))
                

        return drops,drop_width


    def rain_process(self,image, rain_drops, drop_color,_drop_width):
        imshape = image.shape  
        image_t = image.copy()
        for idx,rain_drop in enumerate(rain_drops):
            pt1 = (rain_drop[0], rain_drop[1])
            pt2 = (rain_drop[2], rain_drop[3])
            drop_width = _drop_width[idx]
            cv2.line(image_t, pt1, pt2, drop_color, drop_width)

        return image_t


    def add_rain(self,image, slant=None, drop_color=(200,200,200)):
        imshape = image.shape
        if slant is None:
            slant = np.random.randint(-10, 10) # generate random slant if no slant value is given
        rain_drops = self.generate_random_lines(imshape, slant)
        output = self.rain_process(image, rain_drops, drop_color)

        return output

    def add_rain_step(self,image, slant=None, drop_color=(200,200,200),step_ratio=1.0,return_slant=False):
        imshape = image.shape[:2]
        #imshape = image.shape

            
        if slant is None:
            slant = np.random.randint(-10, 10) # generate random slant if no slant value is given
            
        if not imshape==self.imshape:
            self.rain_drops, self.rain_width = self.generate_random_lines(self.imshape, slant)
            print('self.imshape = %s, imshape = %s.'%(str(self.imshape),str(imshape)))
            

        
        rain_drops = self.rain_drops
        rain_width = self.rain_width
        len_rain = len(rain_drops)
        try:
            final_rains = rain_drops[:int(step_ratio*len_rain)]
            final_rain_width = rain_width[:int(step_ratio*len_rain)]
        except:
            print(rain_width)
            print(len(rain_width))
            print(rain_drops)
            print(len(rain_drops))

            print(step_ratio)
            print(len(step_ratio))

            print(len_rain)
            huidsa
            
            
        output = self.rain_process(image, final_rains, drop_color,final_rain_width)
        if return_slant:
            return output,slant
        else:
            return output



    def generate_spot_light_mask(self,
                                 mask_size,
                                 position=None,
                                 max_brightness=255,
                                 min_brightness=0,
                                 mode="gaussian",
                                 linear_decay_rate=None,
                                 speedup=False,
                                 reverse=False):
        """
        Generate decayed light mask generated by spot light given position, direction. Multiple spotlights are accepted.
        Args:
            mask_size: tuple of integers (w, h) defining generated mask size
            position: list of tuple of integers (x, y) defining the center of spotlight light position,
                      which is the reference point during rotating, format=[(w1,h1), (w2,h2)]
            max_brightness: integer that max brightness in the mask
            min_brightness: integer that min brightness in the mask
            mode: the way that brightness decay from max to min: linear or gaussian
            linear_decay_rate: only valid in linear_static mode. Suggested value is within [0.2, 2]
            speedup: use `shrinkage then expansion` strategy to speed up vale calculation
            reverse: center point is dark or bright (default: bright)
        Return:
            light_mask: ndarray in float type consisting value from max_brightness to min_brightness. If in 'linear' mode
                        minimum value could be smaller than given min_brightness.
        """
        if position is None:
            position = [(random.randint(0, mask_size[0]), random.randint(0, mask_size[1]))]
        if linear_decay_rate is None:
            if mode == "linear_static":
                linear_decay_rate = random.uniform(0.25, 1)
        assert mode in ["linear", "gaussian"], \
            "mode must be linear_dynamic, linear_static or gaussian"
        mask = np.zeros(shape=(mask_size[1], mask_size[0]), dtype=np.float32)
        if mode == "gaussian":
            mu = np.sqrt(mask.shape[0]**2+mask.shape[1]**2)
            dev = mu / 3.5
            mask = self._decay_value_radically_norm_in_matrix(mask_size, position, max_brightness, min_brightness, dev)
        mask = np.asarray(mask, dtype=np.uint8)
        # add median blur
        mask = cv2.medianBlur(mask, 5)
        if reverse:
            mask = 255 - mask
        return mask

    def _decay_value_radically_norm_in_matrix(self,mask_size, centers, max_value, min_value, dev):
        """
        _decay_value_radically_norm function in matrix format
        """
        center_prob = norm.pdf(0, 0, dev)
        x_value_rate = np.zeros((mask_size[1], mask_size[0]))
        for center in centers:
            coord_x = np.arange(mask_size[0])
            coord_y = np.arange(mask_size[1])
            xv, yv = np.meshgrid(coord_x, coord_y)
            dist_x = xv - center[0]
            dist_y = yv - center[1]
            dist = np.sqrt(np.power(dist_x, 2) + np.power(dist_y, 2))
            x_value_rate += norm.pdf(dist, 0, dev) / center_prob
        mask = x_value_rate * (max_value - min_value) + min_value
        mask[mask > 255] = 255
        return mask



    def add_spot_light_step(self,image, light_position=None, max_brightness=255, min_brightness=0,
                       mode='gaussian', linear_decay_rate=None, transparency=None,step_ratio=1.0,reverse=None):
        """
        Add mask generated from spot light to given image
        """
        image=copy.deepcopy(image)
        
        if reverse is None:
            reverse=False
            if np.random.random(size=1)>0.5:
                reverse=True
                
        if transparency is None:
            transparency = random.uniform(0.5, 0.85)
            
        transparency = 1-(1-transparency)*step_ratio
        frame = image
        frame = np.asarray(frame, dtype=np.float32)
        height, width, _ = frame.shape
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = self.generate_spot_light_mask(mask_size=(width, height),
                                        position=light_position,
                                        max_brightness=max_brightness,
                                        min_brightness=min_brightness,
                                        mode=mode,
                                        linear_decay_rate=linear_decay_rate,
                                        reverse=reverse,
                                       )
        hsv[:, :, 2] = hsv[:, :, 2] * transparency + mask * (1 - transparency)
        #print(width)
        #print(height)
        #plt.imshow(mask)
        frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        frame[frame > 255] = 255
        frame[frame < 0] = 0
        frame = np.asarray(frame, dtype=np.uint8)
        return frame

    def adjust_gamma(self,image, gamma=None):
        image=copy.deepcopy(image)
        # build a lookup table mapping the pixel values [0, 255] to
        # their adjusted gamma values
        if gamma is None:
            gamma = np.random.randint(40,91) / 100.0
            #gamma = np.random.randint(50,150) / 100.0
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")
        # apply gamma correction using the lookup table
        #print(table)
        return cv2.LUT(image, table)

    def pixelate(self,image, blocks=None,step_ratio=1.0):
        input_image_copy=copy.deepcopy(image)
        # divide the input image into NxN blocks
        st_ratio=0.5+step_ratio*0.5
        if blocks is None:
            blocks = np.random.randint(120,151)

        (h, w) = image.shape[:2]
        #Xblocks=w-(w-blocks)*step_ratio
        #Yblocks=h-(h-blocks)*step_ratio
        Xblocks=blocks*st_ratio+(1-st_ratio)*w
        Yblocks=blocks*st_ratio+(1-st_ratio)*h
        #print(Xblocks)


        xSteps = np.linspace(0, w, int(Xblocks) + 1, dtype="int")
        ySteps = np.linspace(0, h, int(Yblocks) + 1, dtype="int")
        #print(len(xSteps),len(ySteps))

        #print(xSteps)
        #print(h)
        #print(w)
        # loop over the blocks in both the x and y direction
        limited=False
        for i in range(1, len(ySteps)):
            for j in range(1, len(xSteps)):
                # compute the starting and ending (x, y)-coordinates
                # for the current block
                startX = xSteps[j - 1]
                startY = ySteps[i - 1]
                endX = xSteps[j]
                endY = ySteps[i]
                if endX-startX<2 and endY-startY<2:
                    #limited=True
                    #print('%d %d %d %d'%(startX,endX,startY,endY))

                    #print('limited to original image')
                    break

                roi = input_image_copy[startY:endY, startX:endX]
                #print(roi)
                #print('%d %d %d %d'%(startX,endX,startY,endY))
                (B, G, R) = [int(x) for x in cv2.mean(roi)[:3]]

                cv2.rectangle(input_image_copy, (startX, startY), (endX, endY), (B, G, R), -1)
            #if limited:
            #    break
        # return the pixelated blurred image
        if limited:
            return image
        else:
            return input_image_copy
        
    def deresolution(self,image, blocks=None,step_ratio=1.0):
        input_image_copy=copy.deepcopy(image)
        # divide the input image into NxN blocks
        if step_ratio==0:
            st_ratio=0
        else:
            st_ratio=0.5+step_ratio*0.5
            
        (h, w) = image.shape[:2]
        if blocks is None:
            blocks = np.random.randint(int((h+w)/2*0.2),int((h+w)/2*0.3))

        Xblocks=blocks*st_ratio+(1-st_ratio)*w
        Yblocks=blocks*st_ratio+(1-st_ratio)*h

        img = cv2.resize(input_image_copy,(int(Xblocks),int(Yblocks)),interpolation=cv2.INTER_AREA)
        #print(img.shape)
        img = cv2.resize(img,(w, h),interpolation=cv2.INTER_AREA)
        #print(img.shape)

        return img

        
        
    def mix_aug(self, img, step, random=False):
        # img= numpy [255,255,3]
        # step= 0~1
        if random:
            self.random_parameter()
        if self.noiseStepMode=='exp':
            step = self.multi_base*math.pow(self.exp_base, step*100)
        #print(np.max(img))
        #print(np.min(img))
        img = img.astype(np.uint8)
        height, width, _ = img.shape
        pos = [(self.spotlignt_position[0][0]*width,self.spotlignt_position[0][1]*height)]
        #self.pixelate_blocks=60
        img = self.add_rain_step(img,step_ratio=step,slant = self.rain_slant)
        img = self.add_spot_light_step(img,step_ratio=step,light_position=pos,transparency=self.spotlight_transparency,reverse=self.spotlight_reverse)
        #print(1-(1-self.gamma)*step)
        #print('dsadsada')
        img = self.adjust_gamma(img,gamma=1-(1-self.gamma)*step)
        #img = self.pixelate(img,blocks=self.pixelate_blocks,step_ratio=step)

        img = self.deresolution(img,blocks=self.pixelate_blocks,step_ratio=step)

        #img = self.pixelate_cv2(img,blocks=self.pixelate_blocks,step_ratio=step)

        return img
    
    def torchcuda2npy(self, x):    
        return (x.detach().cpu().numpy().transpose([1,2,0])+1)*0.5
    
    def npy2torchcuda(self, x,device):
        return torch.tensor(((x*2)-1).transpose([2,0,1]), device=device)

    def torchcuda2npy_cv2(self, x):    
        return (x.detach().cpu().numpy().transpose([1,2,0]))
    
    def npy2torchcuda_cv2(self, x,device):
        return torch.tensor(x.transpose([2,0,1]), device=device)
    
    def torchcuda2npy_cv2_batch(self, x):    
        return (x.detach().cpu().numpy().transpose([0,2,3,1]))
    
    def npy2torchcuda_cv2_batch(self, x,device):
        return torch.tensor(x.transpose([0,3,1,2]), device=device)


    
    def batch_data_add_licence_aug(self,imgbf,t,random=False,clamp=False):
        
        after_img = torch.full(imgbf.shape,0,dtype=imgbf.dtype,device=imgbf.device)
        for idx, oneimg in enumerate(imgbf):
            img_np = self.torchcuda2npy_cv2(copy.deepcopy(oneimg))
            img_aug = self.mix_aug(img_np*255,t,random=random)
            #print('img_np_max = '+str(np.max(img_np)))
            #print('img_np_min = '+str(np.min(img_np)))
            #img_aug = img_np*255
            #img_np7 = mix_a.mix_aug(img_np*255,t,random=random)




            img_inv = self.npy2torchcuda_cv2(img_aug/255,imgbf.device)
            if clamp:
                img_inv = torch.clamp(img_inv, min=0.0, max=1.0)
                
            if torch.max(img_inv)>1 or torch.min(img_inv)<0:
                print('img_aug_max = '+str(np.max(img_aug)))
                print('img_aug_min = '+str(np.min(img_aug)))
                print('img_inv = '+str(torch.max(img_inv)))
                print('img_inv = '+str(torch.min(img_inv)))

            #print('img_inv_max = '+str(np.max(img_aug/255)))
            #print('img_inv_min = '+str(np.min(img_aug/255)))

            after_img[idx] = img_inv
        return after_img
    
    def compute_pow(self, start, end, n):
        #  C = e^((ln(end) - ln(start)) / n)
        C = math.exp((math.log(end) - math.log(start)) / n)
        return C


