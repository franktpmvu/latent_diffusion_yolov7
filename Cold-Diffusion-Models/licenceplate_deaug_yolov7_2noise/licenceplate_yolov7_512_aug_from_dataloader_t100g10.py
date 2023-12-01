from licenceplate_deaug_pytorch.licenceplate_deaug_pytorch_aug_in_dataloader_2noise import Unet_2timeEmb, GaussianDiffusion, Trainer
import torchvision
import argparse
import sys
import torch
sys.path.append('/data/licence_plate/_yolo/yolov7/')
from models.yolo_with_diffusion import Model_with_diffusion
import yaml

#cfgyolotiny='/data/licence_plate/_yolo/yolov7/cfg/training/yolov7-tiny.yaml'
#hypyolotiny='/data/licence_plate/_yolo/yolov7/cfg/deploy/hyp.scratch.tiny.yaml'
#best_288_state_dict='/data/licence_plate/_yolo/yolov7/runs/train/exp2/weights/best_288_state_dict.pt'
#nc=35
#with open(hypyolotiny) as f:
#    hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps

#model = Model_with_diffusion(cfgyolotiny, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device).half()  # create
#model.load_state_dict(torch.load(best_288_state_dict))
#model.eval()
#print(model.m)

parser = argparse.ArgumentParser()
parser.add_argument('--time_steps', default=50, type=int)
parser.add_argument('--t_steps', default=50, type=int)
parser.add_argument('--g_steps', default=50, type=int)
parser.add_argument('--train_steps', default=700000, type=int)
parser.add_argument('--save_folder', default='./results_AOLP', type=str)
parser.add_argument('--load_path', default=None, type=str)
parser.add_argument('--data_path_1', default='/data/licence_plate/_plate/generated_data/result2/img/', type=str)
parser.add_argument('--data_path_2', default='/data/licence_plate/_plate/generated_data/result3/img/', type=str)
parser.add_argument('--data_path_3', default='', type=str)
parser.add_argument('--aug_routine', default='Default', type=str)
parser.add_argument('--train_routine', default='Final', type=str)
parser.add_argument('--sampling_routine', default='x0_step_down', type=str)
parser.add_argument('--remove_time_embed', action="store_true")
parser.add_argument('--residual', action="store_true")
parser.add_argument('--predict_noise', action="store_true")
parser.add_argument('--loss_type', default='l1', type=str)#l1,l1_with_last_layer
parser.add_argument('--yolomodel', default='/data/licence_plate/_yolo/yolov7/runs/train/exp2/weights/best_288_state_dict.pt', type=str)
parser.add_argument('--yolohyperparam', default='/data/licence_plate/_yolo/yolov7/cfg/deploy/hyp.scratch.tiny.yaml', type=str)
parser.add_argument('--yolocfg', default='/data/licence_plate/_yolo/yolov7/cfg/training/yolov7-tiny.yaml', type=str)
parser.add_argument('--yoloclass', default=35, type=int)
parser.add_argument('--eval_data_path', default='/data/licence_plate/_plate/generated_data/result4/img/', type=str)
parser.add_argument('--eval_data_label_path', default='/data/licence_plate/_plate/generated_data/result4/label.txt', type=str)


args = parser.parse_args()
print(args)

with open(args.yolohyperparam) as f:
    hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps
    
yolomodel = Model_with_diffusion(args.yolocfg, ch=3, nc=args.yoloclass, anchors=hyp.get('anchors')).cuda() # create
yolomodel.load_state_dict(torch.load(args.yolomodel))
yolomodel.eval()
yolomodel.create_subnetwork()
#yolomodel = torch.nn.DataParallel(yolomodel, device_ids=range(torch.cuda.device_count()))

#yolomodel = torch.nn.DataParallel(yolomodel, device_ids=range(torch.cuda.device_count()))

#def preproccess_img(img_path):
#    img = cv2.imread(img_path)
#    img, meta = resize(img, input_size)
#    img = (img / 255.).astype(np.float32)
#    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x512x512
#    img = np.ascontiguousarray(img)
#    img = torch.from_numpy(img).to(device)
#    img = torch.unsqueeze(img, 0)
#    img = img.half()
#    return img, meta

#img, meta = preproccess_img(img_path)
#with torch.no_grad():
#ret2_bf,y = yolomodel.forward_submodel(img,yolomodel.before_diffusion_model,output_y=True)
#ret2 = yolomodel.forward_submodel(ret2_bf,yolomodel.after_diffusion_model,init_y=y)


model = Unet_2timeEmb(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    channels=128,
    with_time_emb=not(args.remove_time_embed),
    residual=args.residual,
    predict_noise=args.predict_noise
).cuda()

diffusion = GaussianDiffusion(
    model,
    image_size = 64,
    device_of_kernel = 'cuda',
    channels = 128,
    timesteps = args.time_steps,        # number of steps
    t_steps = args.t_steps,        # number of steps
    g_steps = args.g_steps,        # number of steps
    loss_type = args.loss_type,                   # L1 or L2
    aug_routine=args.aug_routine,
    train_routine = args.train_routine,
    sampling_routine = args.sampling_routine,
    yolomodel=yolomodel
).cuda()

import torch
diffusion = torch.nn.DataParallel(diffusion, device_ids=range(torch.cuda.device_count()))

if not args.data_path_3 is '':
    datasets=[args.data_path_1,args.data_path_2,args.data_path_3]
else:
    datasets=[args.data_path_1,args.data_path_2]

trainer = Trainer(
    diffusion,
    datasets,
    image_size = 512,
    train_batch_size = 32,
    eval_batch_size = 32,
    train_lr = 2e-5,
    train_num_steps = args.train_steps, # total training steps
    gradient_accumulate_every = 2,      # gradient accumulation steps
    ema_decay = 0.995,                  # exponential moving average decay
    fp16 = False,                       # turn on mixed precision training with apex
    results_folder = args.save_folder,
    load_path = args.load_path,
    dataset = 'AOLP',
    eval_data_folder = args.eval_data_path,
    eval_data_label_folder = args.eval_data_label_path
)

trainer.train()
