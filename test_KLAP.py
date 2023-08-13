# import package
from glob import glob
import numpy as np
import torch
import torch.nn as nn
import time
import os
from torch.utils.data import DataLoader
from colorama import Style, Fore, Back
from tensorboardX import SummaryWriter
import argparse
from tqdm import tqdm
import random
import torch.nn.functional as F
import json
from mask import Masker_color

# import file
from utils.averageMeter import *
from utils.warmup_scheduler import GradualWarmupScheduler
from utils.utils import *
from utils import pytorch_ssim
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

from torchvision.utils import save_image

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='models.NAFAIGNet.NAFNet')
parser.add_argument('--dataset_train', type=str, default='utils.dataset_mn.DatasetForTrain_sup_self')
parser.add_argument('--dataset_valid', type=str, default='utils.dataset_mn.DatasetForTest_sup_self')

parser.add_argument('--save-dir', type=str, required=True)
parser.add_argument('--max_iter', type=int, default=20)
parser.add_argument('--warmup-epochs', type=int, default=2)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--lr-min', type=float, default=1e-6)
parser.add_argument('--batch-size', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument("--mode", type=str, default = "bayer, quad, nano, qxq, all")
parser.add_argument('--pre_train', type=str, default='False')
parser.add_argument('--gpu_num', type=str, default='0,1')
parser.add_argument('--lambda_n2s', type=float, default=0.0000001)
parser.add_argument('--lambda_reg', type=float, default=1)
parser.add_argument('--EPOCHS', type=int, default=1000)

args = parser.parse_args()

writer = SummaryWriter(os.path.join(args.save_dir, 'log'))
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num

@torch.no_grad()
def evaluate(model, val_loader, iteration, img_ind, args):
    print(Fore.GREEN + "==> Evaluating")
    print("==> Iteration {}/{}".format(iteration, args.max_iter))
    print("==> Image index {}/{}".format(img_ind+1, len(val_loader)))

    psnr_list, ssim_list = [], []
    model.eval()
    start = time.time()
    pBar = tqdm(val_loader, desc='Evaluating')
    for target, image, dataset_label, _, _ in pBar:
#    for target, image, dataset_label in pBar:
#    for target_images, ori_input, dataset_label, input_images_n2self, input_images_mask in pBar:

        if torch.cuda.is_available():
            image = image.cuda()
            target = target.cuda()

        dataset_label = F.one_hot(dataset_label, num_classes=4).cuda()
        
        pred = model([image,dataset_label.float()])   
        save_image(pred[0,[2,1,0],:,:],args.save_dir+f'/results/Output/image{img_ind}_iter{iteration}_pred.png')

        psnr_list.append(torchPSNR(pred, target).item())
        ssim_list.append(pytorch_ssim.ssim(pred, target).item())

    print("\nResults")
    print("------------------")
    print("PSNR: {:.3f}".format(np.mean(psnr_list)))
    print("SSIM: {:.3f}".format(np.mean(ssim_list)))
    print("------------------")
    print('Costing time: {:.3f}'.format((time.time()-start)/60))
    print('Current time:', time.strftime("%H:%M:%S", time.localtime()))
    print(Fore.RED + "---------------------------------------------------------------" + Style.RESET_ALL)

    global writer
    writer.add_scalars('PSNR', {'val psnr': np.mean(psnr_list)}, iteration)
    writer.add_scalars('SSIM', {'val ssim': np.mean(ssim_list)}, iteration)

    return np.mean(psnr_list), np.mean(ssim_list)





def train_metatest(model, model_clone, train_loader, val_loader, optimizer, scheduler, iteration, criterions,args):
    start = time.time()
    print(Fore.CYAN + "==> Training")
    print("==> Iteration {}/{}".format(iteration, args.max_iter))
    print("==> Learning Rate = {:.6f}".format(optimizer.param_groups[0]['lr']))
    meters = get_meter(num_meters=3)

    criterion_l1 = criterions[0]
    criterion_l1_mean = nn.L1Loss()
    

    model.train()
    model_clone.eval()

    pBar = tqdm(train_loader, desc='Training')

    iteration = 1
    lambda_n2s = args.lambda_n2s
    lambda_reg = args.lambda_reg
    
    m = nn.AvgPool2d(2, stride=2)
    u = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    masker = Masker_color(width = 4, mode='interpolate')
    
    for target_images, ori_input, input_images_n2self, input_images_mask, dataset_lab in pBar:
        # Check whether the batch contains all types of degraded data
        if target_images is None: continue
        # move to GPU
        target_images = target_images.cuda()
        b = target_images.shape[0]
        input_images_n2self = input_images_n2self.cuda()
        input_images_mask = input_images_mask.cuda()
        ori_input = ori_input.cuda()
        dataset_label = F.one_hot(dataset_lab, num_classes=4).cuda()
        dataset_lab[:] = 0
        dataset_label_quad = F.one_hot(dataset_lab, num_classes=4).cuda()

        input_images, mask = masker.mask(input_images_n2self, iteration % (masker.n_masks - 1),input_images_mask)
        mask = mask.detach()
        net_output = model([input_images,dataset_label.float()])   
        
        net_output_detach = model_clone([m(ori_input),dataset_label_quad.float()]).detach()
        loss_regularization = criterion_l1_mean(net_output, u(net_output_detach))      

        loss_n2s = criterion_l1(net_output*mask*input_images_mask, input_images_n2self*mask*input_images_mask)/(torch.sum(mask)*b)
        total_loss = (lambda_n2s * loss_n2s) + lambda_reg * loss_regularization    

        
        optimizer.zero_grad()
        ####
        original_state_dict = {}
        for k, v in model.named_parameters():
            original_state_dict[k] = v.clone()

        #####

        total_loss.backward()
        optimizer.step()
        #####
        updated_state_dict = {}
        for k, v in model.named_parameters():
            updated_state_dict[k] = v.clone()
        #####
        new_state_dict = {}
        for key, _ in original_state_dict.items():
            if key.find('transformer') >= 0:
                new_state_dict[key] = updated_state_dict[key].clone()
            else:
                new_state_dict[key] = original_state_dict[key].clone()

        model.load_state_dict(new_state_dict, strict=True)
        #####
        meters = update_meter(meters, [total_loss.item()])
        pBar.set_postfix({'loss': '{:.3f}'.format(meters[0].avg)})

        iteration = iteration+1
    print("\nResults")
    print("------------------")
    print("Total loss: {:.3f}".format(meters[0].avg))
    print("------------------")
    print('Costing time: {:.3f}'.format((time.time()-start)/60))
    print('Current time:', time.strftime("%H:%M:%S", time.localtime()))
    print(Fore.RED + "---------------------------------------------------------------" + Style.RESET_ALL)

    global writer
    writer.add_scalars('loss', {'train total loss': meters[0].avg}, iteration)

    writer.add_scalars('lr', {'Model lr': optimizer.param_groups[0]['lr']}, iteration)

    
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def mask_thr(faig_value,ratio = 20):
    weights = np.zeros((0,1))
    for key in faig_value.item().keys():

        weight = np.reshape(faig_value.item()[key],(-1,1))
        if weight.shape != (1,1):
            weights = np.concatenate([weights,weight],0)
    arg_weights = np.sort(weights.copy(),axis=0)
    thr_value = arg_weights[-int(weights.shape[0]//ratio)]
    return thr_value
def mask_generator(faig_value_bayer,faig_value_quad,faig_value_nano,faig_value_qxq,ratio):
    print("Mask ratio : ",ratio)
    thr_value_bayer = mask_thr(faig_value_bayer,ratio=ratio)
    thr_value_quad = mask_thr(faig_value_quad,ratio=ratio)
    thr_value_nano = mask_thr(faig_value_nano,ratio=ratio)
    thr_value_qxq = mask_thr(faig_value_qxq,ratio=ratio)

    mask_bayer = {}
    mask_quad = {}
    mask_nano = {}
    mask_qxq = {}
    
    for key in faig_value_bayer.item().keys():
        if faig_value_bayer.item()[key].shape != ():
            weight_bayer = torch.tensor((np.expand_dims(np.expand_dims(faig_value_bayer.item()[key],-1),-1)>thr_value_bayer).astype(np.float32))
            weight_quad = torch.tensor((np.expand_dims(np.expand_dims(faig_value_quad.item()[key],-1),-1)>thr_value_quad).astype(np.float32))
            weight_nano = torch.tensor((np.expand_dims(np.expand_dims(faig_value_nano.item()[key],-1),-1)>thr_value_nano).astype(np.float32))
            weight_qxq = torch.tensor((np.expand_dims(np.expand_dims(faig_value_qxq.item()[key],-1),-1)>thr_value_qxq).astype(np.float32))
            
            mask_bayer[key] = torch.unsqueeze(weight_bayer.cuda(),0)
            mask_quad[key] = torch.unsqueeze(weight_quad.cuda(),0)
            mask_nano[key] = torch.unsqueeze(weight_nano.cuda(),0)
            mask_qxq[key] = torch.unsqueeze(weight_qxq.cuda(),0)
    
    return mask_bayer,mask_quad,mask_nano,mask_qxq

def main():
    # Set up random seed
    random_seed = 19820522
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    print(Back.WHITE + 'Random Seed: {}'.format(random_seed) + Style.RESET_ALL)
    print(Fore.RED + "---------------------------------------------------------------" + Style.RESET_ALL)
    print("Batch size",args.batch_size)


    # tensorboard
    os.makedirs(args.save_dir, exist_ok=True)
    
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.save_dir+'/results', exist_ok=True)
    os.makedirs(args.save_dir+'/results/Output', exist_ok=True)
    
    
    with open(os.path.join(args.save_dir, 'args.json'), 'w') as fp:
        json.dump(vars(args), fp, indent=4)


    # get the net and datasets function
    net_func = get_func(args.model)
    # Prepare the Model
    model = net_func().cuda()
    model = nn.DataParallel(model)

    model_clone = net_func().cuda()
    model_clone = nn.DataParallel(model_clone)

    total_param = count_parameters(model)
    print("Total Prameter : ",total_param)
    
    psnr_list = []
    ssim_list = []
    for img_ind in range(args.EPOCHS):
        dataset_train_func = get_func(args.dataset_train)
        dataset_valid_func = get_func(args.dataset_valid)
        print(Back.RED + 'Using Model: {}'.format(args.model) + Style.RESET_ALL)
        print(Back.RED + 'Using Dataset for Train: {}'.format(args.dataset_train) + Style.RESET_ALL)
        print(Back.RED + 'Using Dataset for Valid: {}'.format(args.dataset_valid) + Style.RESET_ALL)
        print(Fore.RED + "---------------------------------------------------------------" + Style.RESET_ALL)

        checkpoint = torch.load(args.pre_train)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        model_clone.load_state_dict(checkpoint['state_dict'], strict=True)


        # load data loader
        train_dataset = dataset_train_func(args.mode, img_ind)
        val_dataset = dataset_valid_func(args.mode, img_ind)
#        train_dataset = dataset_train_func(args.mode)
#        val_dataset = dataset_valid_func(args.mode)
        
        train_loader = DataLoader(dataset=train_dataset, num_workers=args.num_workers, batch_size=args.batch_size,
                                    drop_last=True,  shuffle=True)
        val_loader = DataLoader(dataset=val_dataset, num_workers=args.num_workers, batch_size=1, drop_last=False, shuffle=False)
        print(Style.BRIGHT + Fore.YELLOW + "# Training data / # Val data:" + Style.RESET_ALL)
        print(Style.BRIGHT + Fore.YELLOW + '{} / {}'.format(len(train_dataset), len(val_dataset)) + Style.RESET_ALL)
        print(Fore.RED + "---------------------------------------------------------------" + Style.RESET_ALL)


        print("check ----------------------------")

        # prepare the loss function
        criterions = nn.ModuleList([nn.L1Loss(reduction ="sum")]).cuda()



        # prepare the optimizer and scheduler
        linear_scaled_lr = args.lr
        print("linear_scaled_lr : ",linear_scaled_lr)
        optimizer = torch.optim.Adam([{'params': model.parameters()}], 
                                        lr=linear_scaled_lr, betas=(0.9, 0.999), eps=1e-8)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.max_iter - args.warmup_epochs, eta_min=args.lr_min)
        scheduler.step()
        print("check ----------------------------")


        # Start training pipeline
        start_iter = 1 #checkpoint['epoch']
        top_k_state = []
        ssims =[]
        psnrs =[]    
        print(Fore.GREEN + "Model would be saved on '{}'".format(args.save_dir) + Style.RESET_ALL)
        for iteration in range(args.max_iter+1):
            psnr, ssim = evaluate(model, val_loader, iteration,img_ind,args)
            train_metatest(model,model_clone, train_loader,val_loader, optimizer, scheduler, iteration, criterions,args)
            psnrs.append(psnr)
            ssims.append(ssim)
        np.save(args.save_dir+f'/results/image{img_ind}_iter{iteration}_pred.npy', psnrs)
        
        if img_ind == len(val_dataset)-1:
            break

if __name__ == '__main__':
    main()

