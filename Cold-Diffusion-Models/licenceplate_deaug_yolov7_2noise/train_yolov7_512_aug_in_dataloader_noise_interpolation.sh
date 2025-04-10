#python licenceplate_yolov7_1024_aug_from_dataloader.py --time_steps 100 --aug_routine 'Default' --save_folder './latent_yolov7_1024_aug_from_dataloader_train' --load_path 'latent_yolov7_1024_aug_from_dataloader_train_8gpu/model.pt'

#python licenceplate_yolov7_512_aug_from_dataloader_image.py --time_steps 100 --aug_routine 'Default' --save_folder './latent_yolov7_1024_2loss_image_train' --loss_type 'l1_with_last_layer' #--load_path './latent_yolov7_1024_latentandyolo_train/model.pt'

#python licenceplate_yolov7_512_aug_from_dataloader_image.py --time_steps 100 --aug_routine 'Default' --save_folder './yolov7_512_2loss_image_train' --loss_type 'l1_with_last_layer' --data_path_1 '/data/licence_plate/_plate/synthesis/result_350k/' --data_path_2 '' --data_path_3 '' --yolomodel '/data/licence_plate/_yolo/yolov7/static_model/last_genbg_79_state_dict.pt' --predict_noise --load_path './yolov7_512_2loss_image_train/model.pt'


python licenceplate_yolov7_512_aug_from_dataloader_t100g10.py --time_steps 100 --t_steps 100 --g_steps 10 --aug_routine 'Default' --save_folder './yolov7_512_2loss_interpolation_train' --loss_type 'l1_with_last_layer' --data_path_1 '/data/licence_plate/_plate/synthesis/result_350k/' --data_path_2 '' --data_path_3 '' --yolomodel '/data/licence_plate/_yolo/yolov7/static_model/last_genbg_79_state_dict.pt' --predict_noise --interpolate_noise #--load_path './yolov7_512_2loss_image_train/model.pt'
