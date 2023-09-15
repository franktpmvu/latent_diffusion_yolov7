#python celebA_128.py --time_steps 4 --resolution_routine 'Incremental_factor_2' --save_folder './celebA_4_steps_fac2_train'
#python celebA_test.py --time_steps 4 --train_routine 'Final' --sampling_routine 'x0_step_down' --resolution_routine 'Incremental_factor_2' --save_folder './celebA_test' --load_path './celebA_final_ckpt/model.pt' --test_type 'test_fid_distance_decrease_from_manifold'

#python AOLP_LE_train_256.py --time_steps 4 --resolution_routine 'Incremental_factor_2' --save_folder './AOLP_4_steps_fac2_train'


#python AOLP_LE_test_256.py --time_steps 4 --train_routine 'Final' --sampling_routine 'x0_step_down' --resolution_routine 'Incremental_factor_2' --save_folder './AOLP_test_real3' --load_path './resolution_model_40k.pt' --test_type train_data --data_path '/data/licence_plate/_plate/real_data/ytb_img3/img/' --sample_steps 3 --wonoise

#python AOLP_LE_test_256.py --time_steps 4 --train_routine 'Final' --sampling_routine 'x0_step_down' --resolution_routine 'Incremental_factor_2' --save_folder './AOLP_test_real2' --load_path './resolution_model_40k.pt' --test_type train_data --data_path '/data/licence_plate/_plate/real_data/ytb_img2/img/' --sample_steps 3 --wonoise

#python AOLP_LE_test_256.py --time_steps 4 --train_routine 'Final' --sampling_routine 'x0_step_down' --resolution_routine 'Incremental_factor_2' --save_folder './AOLP_test_real1' --load_path './resolution_model_40k.pt' --test_type train_data --data_path '/data/licence_plate/_plate/real_data/ytb_img/img/' --sample_steps 3 --wonoise

#python AOLP_LE_test_256.py --time_steps 4 --train_routine 'Final' --sampling_routine 'x0_step_down' --resolution_routine 'Incremental_factor_2' --save_folder './AOLP_test_seen' --load_path './resolution_model_40k.pt' --test_type train_data --data_path '/data/licence_plate/_plate/AOLP/LE/train/jpeg/'

#python AOLP_LE_test_256.py --time_steps 4 --train_routine 'Final' --sampling_routine 'x0_step_down' --resolution_routine 'Incremental_factor_2' --save_folder './AOLP_test_unseen' --load_path './resolution_model_40k.pt' --test_type train_data --data_path '/data/licence_plate/_plate/AOLP/LE/test/jpeg/'

python AOLP_LE_test_256.py --time_steps 4 --train_routine 'Final' --sampling_routine 'x0_step_down' --resolution_routine 'Incremental_factor_2' --save_folder './AOLP_test_mix' --load_path './resolution_model_40k.pt' --test_type train_data --data_path '/data/licence_plate/_plate/real_data/AOLP_test/img'