#python licenceplate_yolov7_1024_aug_from_dataloader.py --time_steps 100 --aug_routine 'Default' --save_folder './latent_yolov7_1024_aug_from_dataloader_train' --load_path 'latent_yolov7_1024_aug_from_dataloader_train_8gpu/model.pt'

#python licenceplate_yolov7_512_aug_from_dataloader_t100g10.py --time_steps 100 --t_steps 100 --g_steps 10 --aug_routine 'Default' --save_folder './latent_yolov7_1024_2loss_2noise_train_t100g10' --loss_type 'l1_with_last_layer' --load_path './latent_yolov7_1024_2loss_2noise_train_t100g10/model.pt'


python licenceplate_yolov7_512_aug_from_dataloader_t100g10.py --time_steps 100 --t_steps 100 --g_steps 10 --aug_routine 'Default' --save_folder './latent_yolov7randomgen79_512_2loss_2noise_train_t100g10_pnoise' --loss_type 'l1_with_last_layer' --data_path_1 '/data/licence_plate/_plate/synthesis/result_350k/' --data_path_2 '' --predict_noise --yolomodel '/data/licence_plate/_yolo/yolov7/static_model/last_genbg_79_state_dict.pt' --load_path './latent_yolov7randomgen79_512_2loss_2noise_train_t100g10_pnoise/model.pt'
#--data_path_2   '/data/frank/licence_plate/_plate/cityscapes/leftImg8bit/all/' 
