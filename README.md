official version of License plate recognition in low quality image by using Latent Diffusion YOLOv7
https://ieeecai.org/2024/wp-content/pdfs/540900a842/540900a842.pdf


steps:
1. training a yolov7 model from "license_plate/_yolo/yolov7/train_yolov7_tiny.sh"
2. using the output state_dict to training a latent diffusion by "license_plate/Cold-Diffusion-Models/licenceplate_deaug_yolov7_2noise/train_yolov7_512_aug_in_dataloader_2loss_image_space.sh"
   
