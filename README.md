official version of License plate recognition in low quality image by using Latent Diffusion YOLOv7
https://ieeecai.org/2024/wp-content/pdfs/540900a842/540900a842.pdf


steps:
1. training a yolov7 model from "license_plate/_yolo/yolov7/train_yolov7_tiny.sh"
2. training a image space diffusion: using the output state_dict to training a latent diffusion by "license_plate/Cold-Diffusion-Models/licenceplate_deaug_yolov7_2noise/train_yolov7_512_aug_in_dataloader_2loss_image_space.sh"
3. training a latent space diffusion: "license_plate/Cold-Diffusion-Models/licenceplate_deaug_yolov7_2noise/train_yolov7_512_aug_in_dataloader_2loss_t100_g10.sh" (original, from image(=x) to feature(=z), latent diffusion in z, the noise from image space t(rain etc.) and from latent space g(gaussian))
4. interpolate noise : "license_plate/Cold-Diffusion-Models/licenceplate_deaug_yolov7_2noise/train_yolov7_512_aug_in_dataloader_noise_interpolation.sh" (add interpolation noise in latent space, get x interpolation from x_start latent to x_blur_latent by g_step, create x_interp and diffusion learned from x_interp to x_start_latent)
 
   
