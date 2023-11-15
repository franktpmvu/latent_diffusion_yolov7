#python licenceplate_yolov7_1024_aug_from_dataloader.py --time_steps 100 --aug_routine 'Default' --save_folder './latent_yolov7_1024_aug_from_dataloader_train' --load_path 'latent_yolov7_1024_aug_from_dataloader_train_8gpu/model.pt'

#python licenceplate_yolov7_512_aug_from_dataloader_t100g10.py --time_steps 100 --t_steps 100 --g_steps 10 --aug_routine 'Default' --save_folder './test' --loss_type 'l1_with_last_layer' --data_path_3 '/data/licence_plate/_plate/cityscapes/leftImg8bit/all/' --residual --load_path './latent_yolov7_1024_2loss_2noise_train_t100g10_cityscapes_res/model_best_150000_0.820.pt'

#python licenceplate_yolov7_512_aug_from_dataloader_t100g10.py --time_steps 100 --t_steps 100 --g_steps 10 --aug_routine 'Default' --save_folder './latent_yolov7_clean_512_2loss_2noise_train_t100g10_res_cityscapes' --loss_type 'l1_with_last_layer' --data_path_3 '/data/licence_plate/_plate/cityscapes/leftImg8bit/all/' --residual --yolomodel '/data/licence_plate/_yolo/yolov7/runs/train/exp5/best_726_state_dict.pt' #--load_path './latent_yolov7_1024_2loss_2noise_train_t100g10_cityscapes/model_best_270000_0.766.pt'


#python licenceplate_yolov7_512_aug_from_dataloader_t100g10.py --time_steps 100 --t_steps 100 --g_steps 10 --aug_routine 'Default' --save_folder './latent_yolov7_512_2loss_2noise_train_t100g10_res_cityscapes_exp' --loss_type 'l1_with_last_layer' --data_path_3 '/data/licence_plate/_plate/cityscapes/leftImg8bit/all/' --residual #--load_path './latent_yolov7_1024_2loss_2noise_train_t100g10_cityscapes/model_best_270000_0.766.pt'

python licenceplate_yolov7_512_aug_from_dataloader_t100g10.py --time_steps 100 --t_steps 10 --g_steps 20 --aug_routine 'Default' --save_folder './latent_yolov7_512_2loss_2noise_train_t10g20_res_cityscapes' --loss_type 'l1_with_last_layer' --data_path_3 '/data/licence_plate/_plate/cityscapes/leftImg8bit/all/' --residual --load_path './latent_yolov7_512_2loss_2noise_train_t10g20_res_cityscapes/model.pt'
