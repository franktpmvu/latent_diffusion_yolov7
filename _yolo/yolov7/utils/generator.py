import cv2
import numpy as np
import time
import copy
from PIL import Image, ImageEnhance
from os.path import abspath
from matplotlib import pyplot as plt
import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader







def crop_image(image,hwsize=512):
    height, width = image.shape[:2]

    # Resize the image if smaller than 512 in any dimension
    if height < hwsize or width < hwsize:
        scale = hwsize / min(height, width)
        new_height = max(int(height * scale),hwsize)
        new_width = max(int(width * scale),hwsize)
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    # Get the dimensions of the resized image
    resized_height, resized_width = image.shape[:2]
    #print('resized_height, resized_width = %d,%d'%(resized_height, resized_width))
    # Calculate the top left corner of the crop
    # If one of the dimensions is exactly 512, set the crop coordinate for that dimension to 0
    x = 0 if resized_width == hwsize else np.random.randint(0, resized_width - hwsize)
    y = 0 if resized_height == hwsize else np.random.randint(0, resized_height - hwsize)

    # Crop the image
    cropped_image = image[y:y + hwsize, x:x + hwsize]
    return cropped_image

    









def process_images_in_directory(img_dir='/data/yolov7/coco/images/train2017',hwsize=512):
    """
    Create a generator to process each image in a directory into a 512x512x3 numpy array of type np.uint8.

    :param directory_path: Path to the directory containing image files.
    :return: A generator that yields 512x512x3 numpy arrays of type np.uint8 for each image.
    """
    #img_dir = '/data/yolov7/coco/images/train2017'

    def generator():
        # Create a generator for all image files in the directory
        image_files = glob.iglob(os.path.join(img_dir, '*.[pj][np][g]'))
        #print(image_files.shape)
        for image_file in image_files:
            # Read the image
            image = cv2.imread(image_file)
            #image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            height, width = image.shape[:2]

            # Resize the image if smaller than 512 in any dimension
            if height < hwsize or width < hwsize:
                scale = hwsize / min(height, width)
                new_height = max(int(height * scale),hwsize)
                new_width = max(int(width * scale),hwsize)
                image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

            # Get the dimensions of the resized image
            resized_height, resized_width = image.shape[:2]
            #print('resized_height, resized_width = %d,%d'%(resized_height, resized_width))
            # Calculate the top left corner of the crop
            # If one of the dimensions is exactly 512, set the crop coordinate for that dimension to 0
            x = 0 if resized_width == hwsize else np.random.randint(0, resized_width - hwsize)
            y = 0 if resized_height == hwsize else np.random.randint(0, resized_height - hwsize)

            # Crop the image
            cropped_image = image[y:y + hwsize, x:x + hwsize]

            yield cropped_image

    return generator()
    # Example usage
    # a = process_images_in_directory("path_to_directory")
    # image = next(a)
    # Note: Replace "path_to_directory" with the actual directory path.

def cycle(dl):
    while True:
        for data in dl:
            yield data


    


























class plate_generator():
    def __init__(self,augmentor=None, image_size=512):
        ### Basic parameters settings
        ### 1cm = 38px
        cm = 38

        ### Index-0 is parameter of 6-characters-plate
        ### Index-1 is parameter of 7-characters-plate
        plate_width = [32 * cm, 38 * cm]
        plate_height = [15 * cm, 16 * cm]
        
        if augmentor:
            self.augmentor = augmentor
            self.ratio_augmentation=0.5 # % of need augment

        else:
            self.augmentor = None
        self.box_padding = 20   #px
        self.plate_width = plate_width
        self.plate_height = plate_height
        
        self.char_padding = [int(0.018 * plate_width[0]), int(0.013 * plate_width[1])]
        #self.char_padding = [int(0.038 * plate_width[0]), int(0.013 * plate_width[1])]
        self.char_width = [int(0.136 * plate_width[0]), int(0.12 * plate_width[1])]
        self.char_height = [int(0.6 * plate_height[0]), int(0.604 * plate_height[1])]
        self.char_upper_distance = [int(0.232 * plate_height[0]), int(0.236 * plate_height[1])]
        self.dot = [int(0.037 * plate_width[0]), int(0.031 * plate_width[1])]
        
        self.load_path = ['/data/licence_plate/_plate/synthesis/characters/6_characters/', 
                          '/data/licence_plate/_plate/synthesis/characters/7_characters/']
        self.bg_path = ['/data/licence_plate/_plate/synthesis/transparentBG_mask.png', 
                        '/data/licence_plate/_plate/synthesis/transparentBG2_mask.png']
        
        self.classes = ['0','1','2','3','4','5','6','7','8','9',
                'A','B','C','D','E','F','G','H','J','K','L','M','N','P','Q','R','S','T','U','V','W','X','Y','Z']
        self.classes_dict= {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'A': 10, 'B': 11, 'C': 12, 'D': 13, 'E': 14, 'F': 15, 'G': 16, 'H': 17, 'J': 18, 'K': 19, 'L': 20, 'M': 21, 'N': 22, 'P': 23, 'Q': 24, 'R': 25, 'S': 26, 'T': 27, 'U': 28, 'V': 29, 'W': 30, 'X': 31, 'Y': 32, 'Z': 33, 'plate': 34}
        self.words_image_dict={}

        #self.BG_COLOR = 209
        #self.BG_SIGMA = 5
        #self.MONOCHROME = 1
        self.image_size = image_size
        
        self.coco_images = cycle(process_images_in_directory(img_dir='/data/yolov7/coco/images/train2017'))

        self.colours_mode = [(0,0), (1,2), (3,4)]
        self.blank_plate = [cv2.imread(self.bg_path[0]), cv2.imread(self.bg_path[1])]

        

    def get_plate(self, bg_image=None):
        ## random choose 
        #idx = np.random.randint(3)
        #mode1, mode2 = self.colours_mode[idx]
        #start_time = time.time()

        mode1  = np.random.randint(5)
        mode2  = np.random.randint(5)
        ##
        
        
        
        # create base plate
        #blank_time = time.time()
        #blank_plate = self.blank_plate

        #_blank = blank_plate[0].copy()
        #img1, sheet1 = self.basic_plate_allrandom(_blank)
        #_blank = blank_plate[0].copy()
        #img2, sheet2 = self.basic_plate_allrandom(_blank)
        
        img1, sheet1 = self.basic_plate_allrandom()
        img2, sheet2 = self.basic_plate_allrandom()

        #print("Total time taken blank_time = {} seconds".format( time.time() - blank_time))  # print total computation time

        
        # adj color
        #adj_time = time.time()

        #img1 = self.adjust_colour(img1, mode1)
        #img2 = self.adjust_colour(img2, mode2)
        #print("Total time taken adj_time = {} seconds".format( time.time() - adj_time))  # print total computation time
        
        # adj color
        #adj2_time = time.time()

        #img1_ = self.adjust_colour_(img1, mode1)
        #img2_ = self.adjust_colour_(img2, mode2)
        #print("Total time taken adj2_time = {} seconds".format( time.time() - adj2_time))  # print total computation time

        # adj color
        #adj3_time = time.time()

        img1 = self.adjust_colour__(img1, mode1)
        img2 = self.adjust_colour__(img2, mode2)
        #print("Total time taken adj3_time = {} seconds".format( time.time() - adj3_time))  # print total computation time

        
        # resize
        #resize_time = time.time()
        img1, sheet1 = self.resize_plate(img1, sheet1)
        img2, sheet2 = self.resize_plate(img2, sheet2)
        #print("Total time taken resize_time = {} seconds".format( time.time() - resize_time))  # print total computation time

        # put plate on image without overlapping
        #mix_plates_time = time.time()

        shape1 = img1.shape[0:2]
        shape2 = img2.shape[0:2]
        while True:
            _iplate1 = img1.copy()
            _iplate2 = img2.copy()
            blank1, offset1 = self.paste_to_black_BG(_iplate1,nw=self.image_size,nh=self.image_size)
            blank2, offset2 = self.paste_to_black_BG(_iplate2,nw=self.image_size,nh=self.image_size)
            dst1, M1 = self.perspective_transform_with_angle(shape1, blank1, offset1)
            dst2, M2 = self.perspective_transform_with_angle(shape2, blank2, offset2)
            if not(self.intersect_optimized(dst1, dst2)):
                break
                
                
        # merge image
        fimg = self.merge_img(dst1, dst2)
        #print("Total time taken mix_plates_time = {} seconds".format( time.time() - mix_plates_time))  # print total computation time

        
        # update label
        #update_label_time = time.time()

        fsheet = self.update_log(sheet1, sheet2, offset1, offset2, M1, M2)
        #print("Total time taken update_label_time = {} seconds".format( time.time() - update_label_time))  # print total computation time

        # put plate on bg
        #put_plate_on_bg_time = time.time()

        if not bg_image is None:
            rnd = crop_image(bg_image, hwsize = self.image_size)
        else:
            #rnd = next(self.coco_images)
            rnd = self.random_bg(image_size=self.image_size)
        
        #fimg = self.match_contrast_and_brightness(fimg, rnd) #have some bug will change color of words
        #print("Total time taken put_plate_on_bg_time = {} seconds".format( time.time() - put_plate_on_bg_time))  # print total computation time

        
        
        # add noises
        #add_noise_time = time.time()
        
        
        nimg = self.change_bg(fimg, rnd)
        nimg = self.add_texture(nimg)
        nimg = self.add_blur(nimg)
        nimg = self.contrast_brightness(nimg)
        
        #print("Total time taken add_noise_time = {} seconds".format( time.time() - add_noise_time))  # print total computation time

        
        if self.augmentor:
            need_aug = np.random.randint(0,99)
            if 0.01*need_aug>=self.ratio_augmentation:
                #add_aug_time = time.time()

                step = np.random.randint(0,99)
                nimg = self.augmentor.mix_aug(nimg, step*0.01, random=True)
                #print("Total time taken add_aug_time = {} seconds".format( time.time() - add_aug_time))  # print total computation time
        

            
        #msg = self.get_msg(fsheet)  # Attention! The invald labels are bot be filtered!
        label = self.get_xywh(fsheet)  # Attention! The invald labels are bot be filtered!
        #print("Total time taken all = {} seconds".format( time.time() - start_time))  # print total computation time

        return nimg, label
    
    def randColor(self):
        return np.array([np.random.random(), np.random.random(), np.random.random()]).reshape((1, 1, 3))
    
    def safeDivide(self, a, b):
        return np.divide(a, np.maximum(b, 0.001))

    
    def random_bg(self,image_size=512):
        def getX(): return xArray
        def getY(): return yArray

        dX, dY = image_size, image_size
        xArray = np.linspace(0.0, 1.0, dX).reshape((1, dX, 1))
        yArray = np.linspace(0.0, 1.0, dY).reshape((dY, 1, 1))  

        functions = [(0, self.randColor),
                    (0, getX),
                    (0, getY),
                    (1, np.sin),
                    (1, np.cos),
                    (2, np.add),
                    (2, np.subtract),
                    (2, np.multiply),
                    (2, self.safeDivide)]
        depthMin = 2
        depthMax = 10

        def buildImg(depth = 0):
            funcs = [f for f in functions if
                        (f[0] > 0 and depth < depthMax) or
                        (f[0] == 0 and depth >= depthMin)]
            idx = np.random.choice(len(funcs))
            nArgs, func = funcs[idx]
            args = [buildImg(depth + 1) for n in range(nArgs)]
            return func(*args)

        img = buildImg()
        while (img.shape[2] != 3) or ((dX/img.shape[0]) != 1) or ((dY/img.shape[1]) != 1):
            img = buildImg()
        #print((dX / img.shape[0], dY / img.shape[1], 3 / img.shape[2]))
        # Ensure it has the right dimensions, dX by dY by 3
        #img = np.tile(img, (dX / img.shape[0], dY / img.shape[1], 3 / img.shape[2]))

        # Convert to 8-bit, send to PIL and save
        img8Bit = np.uint8(np.rint(img.clip(0.0, 1.0) * 255.0))

        return np.array(img8Bit)

    
    
    
    def match_contrast_and_brightness(self, src, dst):
        """
        Match the contrast and brightness of two images, leaving black pixels in src unchanged.

        Parameters:
        - src: NumPy array (image to be adjusted)
        - dst: NumPy array (reference image)

        Returns:
        - adjusted_src: NumPy array (src adjusted to match the contrast and brightness of dst)
        """
        # Mask to identify non-black pixels in src
        mask = np.any(src != [0, 0, 0], axis=-1)

        # Calculate mean and std deviation for non-black pixels in src and for dst
        mean_src, std_src = src[mask].mean(), src[mask].std()
        mean_dst, std_dst = dst.mean(), dst.std()

        # Adjust contrast and brightness only for non-black pixels
        adjusted_src = src.copy()  # Create a copy of the source image
        adjusted_src[mask] = (src[mask] - mean_src) * (std_dst / std_src) + mean_dst
        adjusted_src[mask] = np.clip(adjusted_src[mask], 0, 255)

        # Adjust brightness
        brightness_factor = mean_dst / adjusted_src[mask].mean()
        adjusted_src[mask] = adjusted_src[mask] * brightness_factor
        adjusted_src[mask] = np.clip(adjusted_src[mask], 0, 255)

        return adjusted_src.astype('uint8')

    
    
    def create_texture(self, image, sigma=5, turbulence=2):
        """
        Consequently applies noise patterns to the original image from big to small.

        sigma: defines bounds of noise fluctuations
        turbulence: defines how quickly big patterns will be replaced with the small ones. The lower
        value - the more iterations will be performed during texture generation.
        """
        result = image.astype(float)
        cols, rows, ch = image.shape
        ratio = cols
        while not ratio == 1:
            result += self.create_noise(cols, rows, ratio, sigma=sigma)
            ratio = (ratio // turbulence) or 1
        cut = np.clip(result, 0, 255)
        return cut.astype(np.uint8)

    def create_noise(self, width, height, ratio=1, sigma=5):
        """
        The function generates an image, filled with gaussian nose. If ratio parameter is specified,
        noise will be generated for a lesser image and then it will be upscaled to the original size.
        In that case noise will generate larger square patterns. To avoid multiple lines, the upscale
        uses interpolation.

        :param ratio: the size of generated noise "pixels"
        :param sigma: defines bounds of noise fluctuations
        """
        mean = 0
        assert width % ratio == 0, "Can't scale image with of size {} and ratio {}".format(width, ratio)
        assert height % ratio == 0, "Can't scale image with of size {} and ratio {}".format(height, ratio)

        h = int(height / ratio)
        w = int(width / ratio)

        result = np.random.normal(mean, sigma, (w, h, 1))
        if ratio > 1:
            result = cv2.resize(result, dsize=(width, height), interpolation=cv2.INTER_LINEAR)
        return result.reshape((width, height, 1))

    
    def add_noise(self, img, sigma=5):
        """
        Adds noise to the existing image
        """
        width, height, ch = img.shape
        n = self.create_noise(width, height, sigma=sigma)
        img = img + n
        return img.clip(0, 255)

    
    def create_blank_image(self, width=512, height=512, background=209):
        """
        It creates a blank image of the given background color
        """
        img = np.full((height, width, 1), background, np.uint8)
        return img



    def get_msg(self, s):
        ### Create label string
        ### Attention!! The invalid labels are not be filtered!!

        msg = ''
        for _s in s:
            class_num = str(_s['Class'])
            x1, y1 = _s['p1']
            x2, y2 = _s['p2']
            x3, y3 = _s['p3']
            x4, y4 = _s['p4']
            msg = msg + '%s %r,%r %r,%r %r,%r %r,%r\n' % (class_num, x1, y1, x2, y2, x3, y3, x4, y4)

        return msg
    
    def get_xywh(self, s):
        ### Create label xy = center

        msg = []
        for _s in s:
            #class_num = self.classes_dict[str(_s['Class'])]
            
            if _s['Class']=='plate':
                class_num=34
            else:
                class_num=_s['Class']

            x1, y1 = _s['p1']
            x2, y2 = _s['p2']
            x3, y3 = _s['p3']
            x4, y4 = _s['p4']
            #msg = msg + '%s %r,%r %r,%r %r,%r %r,%r\n' % (class_num, x1, y1, x2, y2, x3, y3, x4, y4)
            xmin = np.min((x1,x2,x3,x4))/self.image_size
            xmax = np.max((x1,x2,x3,x4))/self.image_size
            ymin = np.min((y1,y2,y3,y4))/self.image_size
            ymax = np.max((y1,y2,y3,y4))/self.image_size
            w=xmax-xmin
            h=ymax-ymin
            #if class_num ==34: 
            #    print(xmin*self.image_size)
            #    print(xmax*self.image_size)
            #    print(ymin*self.image_size)
            #    print(ymax*self.image_size)
            msg.append([class_num,xmin+w/2,ymin+h/2,w,h])
        return msg

    
    
        
    def contrast_brightness(self, img):
        new_image = np.zeros(img.shape, img.dtype)
        alpha = np.random.randint(70,116)/100  # Simple contrast control
        beta = np.random.randint(0,11)         # Simple brightness control.

        # Do the operation new_image(i,j) = alpha*image(i,j) + beta
        # Instead of these 'for' loops we could have used simply:
        # new_image = cv.convertScaleAbs(image, alpha=alpha, beta=beta)
        new_image = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        #for y in range(img.shape[0]):
        #    for x in range(img.shape[1]):
        #        for c in range(img.shape[2]):
        #            new_image[y,x,c] = np.clip(alpha*img[y,x,c] + beta, 0, 255)

        return new_image
    
    
    def add_texture(self, dst):
        texture_bg = self.add_noise(self.create_texture(self.create_blank_image(background=230), sigma=4), sigma=10)
        #print(texture_bg.shape)
        t_bg = np.repeat(texture_bg, 3, axis=-1)
        #print(t_bg.shape)
        out = (dst + t_bg)/2
        return out

    def add_blur(self, img, size=3):
        blur_img = cv2.GaussianBlur(img,(size,size),0)
        return blur_img

        
    def change_bg(self, dst, img8Bit):
        # Convert img8Bit to BGR if it's in RGB
        #if img8Bit.shape[2] == 3:  # we expect 3 channels
        #    img8Bit = cv2.cvtColor(img8Bit, cv2.COLOR_RGB2BGR)

        # Create a mask where black pixels ([0, 0, 0]) are True
        mask = np.all(dst == [0, 0, 0], axis=-1)

        # Where mask is True, replace dst pixel with img8Bit pixel
        dst[mask] = img8Bit[mask]

        return dst

        
        
    def update_log(self, s1, s2, offset1, offset2, M1, M2):
        ### Update label information
        w1, h1 = offset1
        w2, h2 = offset2
        tags = ['p1', 'p2', 'p3', 'p4']
        #pts1 = np.float32([[offset_w, offset_h], [offset_w + w, offset_h],
        #                   [offset_w, offset_h + h], [offset_w + w, offset_h + h]])
        #print(cv2.perspectiveTransform(np.array(pts1,dtype=np.float32).reshape(-1,1,2),M))

        #print(s1)
        #print(s2)


        for _s in s1:
            for t in tags:
                pre = np.array([[_s[t][0]+w1], [_s[t][1]+h1], [1]])
                post = np.dot(M1, pre)
                _s[t] = [int(post[0]/post[2]), int(post[1]/post[2])]
        for _s in s2:
            for t in tags:
                pre = np.array([[_s[t][0]+w2], [_s[t][1]+h2], [1]])
                post = np.dot(M2, pre)
                _s[t] = [int(post[0]/post[2]), int(post[1]/post[2])]
        #print(s1)
        #print(s2)

        return np.concatenate((s1,s2), axis=0)

    def merge_img(self, img1, img2):
        # Create a mask where img2 pixels are not black
        mask = ~np.all(img2 == [0, 0, 0], axis=-1)

        # Use this mask to replace pixels in img1 with those from img2
        img1[mask] = img2[mask]

        return img1



    def intersect_optimized(self, img1, img2):
        # Create masks where pixels are not black ([0, 0, 0])
        mask1 = ~np.all(img1 == [0, 0, 0], axis=-1)
        mask2 = ~np.all(img2 == [0, 0, 0], axis=-1)

        # Check if there is any overlap between the masks
        intersection = np.any(np.logical_and(mask1, mask2))

        return intersection

    def perspective_transform_with_angle(self, img_shape, blank_image, offset):
        h, w = img_shape
        nh, nw, nch = blank_image.shape
        offset_w, offset_h = offset

        # Random direction for perspective distortion
        direct = np.random.choice([1, -1], size=2)
        # Random factors for perspective distortion
        r1 = np.random.randint(10, 51) / 100
        r2 = np.random.randint(5, 16) / 100
        #r1 = np.random.randint(1, 51) / 100
        #r2 = np.random.randint(1, 25) / 100
        #r1 = np.random.randint(1, 10) / 100
        #r2 = np.random.randint(1, 5) / 100
        # Calculate distortion based on random factors and direction
        dy = r1 * h * direct[0]
        dx = r2 * w * direct[1]

        # Random angle for rotation (in degrees)
        #angle = np.random.uniform(-30, 30)  # Adjust the range as needed
        angle = np.random.uniform(-9, 9)  # Adjust the range as needed
        angle_rad = np.deg2rad(angle)  # Convert to radians

        # Calculate the rotation matrix for the given angle
        rotation_matrix = cv2.getRotationMatrix2D((offset_w + w/2, offset_h + h/2), angle, 1.0)

        # Apply the rotation to the corner points
        def rotate_point(x, y):
            new_x = (x - offset_w - w/2) * rotation_matrix[0][0] + (y - offset_h - h/2) * rotation_matrix[0][1] + offset_w + w/2
            new_y = (x - offset_w - w/2) * rotation_matrix[1][0] + (y - offset_h - h/2) * rotation_matrix[1][1] + offset_h + h/2
            return new_x, new_y

        # Rotate and then apply perspective distortion to corner points
        pts1 = np.float32([[offset_w, offset_h], [offset_w + w, offset_h],
                           [offset_w, offset_h + h], [offset_w + w, offset_h + h]])
        pts2 = np.float32([rotate_point(offset_w, offset_h),
                           rotate_point(offset_w + w + dx, offset_h + dy),
                           rotate_point(offset_w + dx, offset_h + h + dy),
                           rotate_point(offset_w + w + dx, offset_h + h + dy)])

        # Get the perspective transformation matrix
        M = cv2.getPerspectiveTransform(pts1, pts2)
        #print(pts1)
        #print(pts2)
        #print(M)
        #pts1[:,:,2]=1
        #one_=np.ones((4,1))
        #print(one_)
        #pts1 = np.concatenate((pts1,one_),axis=1)
        #print(pts1.reshape(-1,1,3))
        
        #print(cv2.perspectiveTransform(np.array(pts1,dtype=np.float32).reshape(-1,1,2),M))
        # Apply the perspective transformation
        dst = cv2.warpPerspective(blank_image, M, (nw, nh))

        return dst, M

        
    def paste_to_black_BG(self, iresize, nw=512, nh=512):
        ### Create a blank black image and overlay plate on it
        h, w, ch = iresize.shape
        blank_image = np.zeros((nh,nw,3), np.uint8)
        offset_h = np.random.randint(0+nh//8,nh-h-nh//8)
        offset_w = np.random.randint(0+nw//8,nw-w-nw//8)
        blank_image[offset_h:offset_h+h, offset_w:offset_w+w] = iresize

        return blank_image, [offset_w, offset_h]

        
    def resize_plate(self,iplate, sheet):
        h, w, ch = iplate.shape
        r = np.random.randint(5,16)/10
        nw = int(w*r*0.1)
        nh = int(h*r*0.1)
        iresize = cv2.resize(iplate, (nw,nh))
        scaling_x = nw/w
        scaling_y = nh/h
        for _s in sheet:
            _s['p1'] = [int(_s['p1'][0]*scaling_x), int(_s['p1'][1]*scaling_y)]
            _s['p2'] = [int(_s['p2'][0]*scaling_x), int(_s['p2'][1]*scaling_y)]
            _s['p3'] = [int(_s['p3'][0]*scaling_x), int(_s['p3'][1]*scaling_y)]
            _s['p4'] = [int(_s['p4'][0]*scaling_x), int(_s['p4'][1]*scaling_y)]
        #print(sheet)

        return iresize, sheet

    def adjust_colour(self,iplate, mode = 0):
        ### Change the Background into random color
        ### mode = 0 -> (BG, Word): (White, Black)
        ###        1                (White, Green)
        ###        2                (White, Red)
        ###        3                (Green, White)
        ###        4                (Red, White)
        ### original image 255 = background, 54 = words, 0 = edges
        
        if mode < 3:
            colour = np.random.randint(50,180,size=3)   #Black Background
            colour2 = np.random.randint(230,250,size=3) #White Background
            diff = min(255-colour2[0], 255-colour2[1], 255-colour2[2])
            colour2 = colour2 + diff

            iplate[np.where((iplate==[0,0,0]).all(axis=2))] = [colour[0], colour[1], colour[2]]
            iplate[np.where((iplate==[255,255,255]).all(axis=2))] = [colour2[0], colour2[1], colour2[2]]

            if mode == 1:
                green = np.random.randint(50, 101)
                iplate[np.where((iplate==[54,54,54]).all(axis=2))] = [1, green, 1] #BGR
            elif mode == 2:
                red = np.random.randint(100, 151)
                iplate[np.where((iplate==[54,54,54]).all(axis=2))] = [1, 1, red] #

            return iplate
        else:
            white = np.random.randint(230,250,size=3)
            iplate[np.where((iplate==[54,54,54]).all(axis=2))] = [white[0], white[1], white[2]] #BGR

            colour = np.random.randint(50,180,size=3)
            iplate[np.where((iplate==[0,0,0]).all(axis=2))] = [colour[0], colour[1], colour[2]]

            if mode == 3:
                green = np.random.randint(50, 101)
                iplate[np.where((iplate==[255,255,255]).all(axis=2))] = [1, green, 1]
            elif mode == 4:
                red = np.random.randint(100, 151)
                iplate[np.where((iplate==[255,255,255]).all(axis=2))] = [1, 1, red]

            return iplate

    def adjust_colour_(self, iplate, mode=0):
        black_mask = (iplate == [0, 0, 0]).all(axis=2)
        white_mask = (iplate == [255, 255, 255]).all(axis=2)
        word_mask = (iplate == [54, 54, 54]).all(axis=2)

        if mode < 3:
            colour = np.random.randint(50, 180, size=3)
            colour2 = np.random.randint(230, 250, size=3)
            diff = 255 - colour2.max()
            colour2 += diff

            iplate[black_mask] = colour
            iplate[white_mask] = colour2

            if mode == 1:
                green = np.random.randint(50, 101)
                iplate[word_mask] = [1, green, 1]  # BGR for green word
            elif mode == 2:
                red = np.random.randint(100, 151)
                iplate[word_mask] = [1, 1, red]  # BGR for red word

        else:
            white = np.random.randint(230, 250, size=3)
            iplate[word_mask] = white  # BGR for white word

            colour = np.random.randint(50, 180, size=3)
            iplate[black_mask] = colour

            if mode == 3:
                green = np.random.randint(50, 101)
                iplate[white_mask] = [1, green, 1]  # BGR for green background
            elif mode == 4:
                red = np.random.randint(100, 151)
                iplate[white_mask] = [1, 1, red]  # BGR for red background

        return iplate

    def adjust_colour__(self, iplate, mode=0):
        ### Change the Background into random color
        ### mode = 0 -> (BG, Word): (White, Black)
        ###        1                (White, Green)
        ###        2                (White, Red)
        ###        3                (Green, White)
        ###        4                (Red, White)
        ### original image 255 = background, 54 = words, 0 = edges

        black_mask = (iplate == [0, 0, 0]).all(axis=2)
        #white_mask = (iplate == [255, 255, 255]).all(axis=2)
        word_mask = (iplate == [54, 54, 54]).all(axis=2)
        #print(word_mask==True)
        #print(black_mask==True)
        #print(mode)
        if mode < 3:
            colour = np.random.randint(50, 180, size=3)
            colour2 = np.random.randint(230, 250, size=3)
            diff = 255 - colour2.max()
            colour2 += diff
            iplate[:,:,:] = colour2
            iplate[black_mask] = colour

            if mode == 1:
                green = np.random.randint(50, 101)
                iplate[word_mask] = [1, green, 1]  # BGR for green word
                #print(green)

            elif mode == 2:
                red = np.random.randint(100, 151)
                iplate[word_mask] = [1, 1, red]  # BGR for red word
                #print(red)

            elif mode == 0:
                iplate[word_mask] = [54, 54, 54]  # BGR for black word
                #print(54)

            #print(colour)
            #print(colour2)


        else:
            white = np.random.randint(230, 250, size=3)

            colour = np.random.randint(50, 180, size=3)

            if mode == 3:
                green = np.random.randint(50, 101)
                iplate[:,:,:] = [1, green, 1]  # BGR for green background
            elif mode == 4:
                red = np.random.randint(100, 151)
                iplate[:,:,:] = [1, 1, red]  # BGR for red background
                
            iplate[word_mask] = white  # BGR for white word
            iplate[black_mask] = colour

        return iplate

        
        
    def load_image_from_dict(self,path):
        if path in self.words_image_dict:
            img = self.words_image_dict[path]
        else:
            img = cv2.imread(path)
            self.words_image_dict[path] = img
        return img

        
        
    def basic_plate_allrandom(self):
        blank_plate = self.blank_plate

        
        box_padding = self.box_padding
        plate_width = self.plate_width
        plate_height = self.plate_height
        char_padding = self.char_padding
        char_width = self.char_width
        char_height = self.char_height
        char_upper_distance = self.char_upper_distance
        dot = self.dot
        load_path = self.load_path
        classes = self.classes
        now_load_path = load_path[np.random.randint(2)]
        #print(now_load_path)
        
        plate_nums = np.random.randint(6,8,1)
        #plate_nums = plate_nums = np.random.randint(7,8,1)
        mode = plate_nums[0]-6
        iplate = blank_plate[mode].copy()
        #print(mode)
        license = np.random.randint(0,34,size=plate_nums)
        plate_format = np.random.randint(0,2)
        if plate_nums==7:
            if plate_format == 0:
                c_list1 = [license[0],license[1],license[2]]
                c_list2 = [license[3],license[4],license[5],license[6]]
            elif plate_format == 1:
                c_list1 = [license[3],license[4],license[5],license[6]]
                c_list2 = [license[0],license[1],license[2]]
            #print('plate_nums==7')
        if plate_nums==6:
            if plate_format == 0:
                c_list1 = [license[0],license[1]]
                c_list2 = [license[2],license[3],license[4],license[5]]
            elif plate_format == 1:
                c_list1 = [license[2],license[3],license[4],license[5]]
                c_list2 = [license[0],license[1]]


        sheet = []
        oh = 20 + char_upper_distance[mode]
        ow = 20 + char_padding[mode]
        for i in c_list1:
            ipath = load_path[mode] + classes[i] + '.png'

            #img = cv2.imread(ipath)
            img = self.load_image_from_dict(ipath)
            ni = cv2.resize(img,(char_width[mode], char_height[mode]))
            iplate[oh : oh+char_height[mode], ow : ow+char_width[mode]] = ni
            dic = {}
            dic['Class']=i
            dic['p1']=[ow, oh]
            dic['p2']=[ow+char_width[mode], oh]
            dic['p3']=[ow, oh+char_height[mode]]
            dic['p4']=[ow+char_width[mode], oh+char_height[mode]]
            sheet.append(dic)
            ow += char_padding[mode] + char_width[mode]
            
        #cDot = cv2.imread(load_path[mode]+'dot.png')
        cDot = self.load_image_from_dict(load_path[mode]+'dot.png')
        iDot = cv2.resize(cDot,(dot[mode], char_height[mode]))
        h, w, ch = iDot.shape
        iplate[oh : oh+h, ow : ow+w] = iDot
        ow += char_padding[mode] + w

        for i in c_list2:
            ipath = load_path[mode] + classes[i] + '.png'
            #img = cv2.imread(ipath)
            img = self.load_image_from_dict(ipath)
            ni = cv2.resize(img,(char_width[mode], char_height[mode]))
            iplate[oh : oh+char_height[mode], ow : ow+char_width[mode]] = ni
            dic = {}
            dic['Class']=i
            dic['p1']=[ow, oh]
            dic['p2']=[ow+char_width[mode], oh]
            dic['p3']=[ow, oh+char_height[mode]]
            dic['p4']=[ow+char_width[mode], oh+char_height[mode]]
            sheet.append(dic)
            ow += char_padding[mode] + char_width[mode]

        dic = {}
        dic['Class'] = 'plate'
        dic['p1'] = [box_padding, box_padding]
        dic['p2'] = [box_padding+plate_width[mode], box_padding]
        dic['p3'] = [box_padding, box_padding+plate_height[mode]]
        dic['p4'] = [box_padding+plate_width[mode], box_padding+plate_height[mode]]
        sheet.append(dic)
        #print(iplate.shape)
        #print(sheet[-1])
        #print(mode)
        return iplate, sheet