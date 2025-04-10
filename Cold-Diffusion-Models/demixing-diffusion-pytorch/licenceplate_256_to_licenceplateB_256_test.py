#from comet_ml import Experiment
from demixing_diffusion_pytorch.demixing_diffusion_licencepairs_pytorch import Unet, GaussianDiffusion, Trainer
import torchvision
import os
import errno
import shutil
import argparse

def create_folder(path):
    try:
        os.mkdir(path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

def del_folder(path):
    try:
        shutil.rmtree(path)
    except OSError as exc:
        pass


parser = argparse.ArgumentParser()
parser.add_argument('--time_steps', default=50, type=int)
parser.add_argument('--train_steps', default=700000, type=int)
parser.add_argument('--save_folder', default='./results_cifar10', type=str)
parser.add_argument('--data_path_1', default='/data/licence_plate/_plate/generated_data/result2/img/', type=str)
parser.add_argument('--data_path_2', default='/data/licence_plate/_plate/generated_data/result3/img/', type=str)
parser.add_argument('--load_path', default=None, type=str)
parser.add_argument('--train_routine', default='Final', type=str)
parser.add_argument('--sampling_routine', default='default', type=str)
parser.add_argument('--remove_time_embed', action="store_true")
parser.add_argument('--residual', action="store_true")
parser.add_argument('--loss_type', default='l1', type=str)
parser.add_argument('--test_type', default='train_data', type=str)
parser.add_argument('--noise', default=0, type=float)




args = parser.parse_args()
print(args)

#img_path=None
#if 'train' in args.test_type:
#    img_path = args.data_path_start
#elif 'test' in args.test_type:
#    img_path = args.data_path_start



model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    channels=3,
    with_time_emb=not(args.remove_time_embed),
    residual=args.residual
).cuda()

diffusion = GaussianDiffusion(
    model,
    image_size = 256,
    channels = 3,
    timesteps = args.time_steps,   # number of steps
    loss_type = args.loss_type,    # L1 or L2
    train_routine = args.train_routine,
    sampling_routine = args.sampling_routine
).cuda()

import torch
diffusion = torch.nn.DataParallel(diffusion, device_ids=range(torch.cuda.device_count()))

trainer = Trainer(
    diffusion,
    [args.data_path_1,args.data_path_2],
    image_size = 256,
    train_batch_size = 4,
    train_lr = 2e-5,
    train_num_steps = args.train_steps,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    fp16 = False,                       # turn on mixed precision training with apex
    results_folder = args.save_folder,
    load_path = args.load_path,
    dataset = 'train'
)
print([args.data_path_1,args.data_path_2])

if args.test_type == 'train_data':
    trainer.test_from_data('train', s_times=None)
elif args.test_type == 'test_data':
    trainer.test_from_data('test', s_times=None)


#trainer.train()