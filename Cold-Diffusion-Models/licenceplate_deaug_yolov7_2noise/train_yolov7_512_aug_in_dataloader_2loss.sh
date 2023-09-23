#python licenceplate_yolov7_1024_aug_from_dataloader.py --time_steps 100 --aug_routine 'Default' --save_folder './latent_yolov7_1024_aug_from_dataloader_train' --load_path 'latent_yolov7_1024_aug_from_dataloader_train_8gpu/model.pt'

python licenceplate_yolov7_512_aug_from_dataloader.py --time_steps 100 --aug_routine 'Default' --save_folder './latent_yolov7_1024_2loss_2noise_train' --loss_type 'l1_with_last_layer' #--load_path './latent_yolov7_1024_latentandyolo_train/model.pt'
