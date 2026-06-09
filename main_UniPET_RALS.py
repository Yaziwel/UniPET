import os 
os.environ['CUDA_VISIBLE_DEVICES']='0' 
from model.UniPET import UniPET, Discriminator
from evaluation.metrics import Metrics
from data.common import transformPET, dataIO, merge_patch
from data.PET_Data import Train_Data, Eval_Data, DataSampler
from loss.losses import EdgeAwareLoss, CharbonnierLoss
import numpy as np
import pickle
import torch
from torch import nn
from torch.utils.data import DataLoader
import time 
from tqdm import tqdm
import random
from tools import set_seeds, mkdir 
from torch.nn import functional as F 

def get_textured_region(data, thresh=0.001):
    """
    input shape: [N, C, D, H, W]

    var(x) = E[x^2] - E[x]^2
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)

    E_x2 = F.avg_pool3d(data ** 2, kernel_size=3, stride=1, padding=1)
    Ex_2 = F.avg_pool3d(data, kernel_size=3, stride=1, padding=1) ** 2
    var = E_x2 - Ex_2

    out = (var >= thresh).to(data.dtype)

    return out

def GradientPenalty(discriminator, real, fake, lambda_=10):
    """
    gradient penalty
    """
    assert real.size() == fake.size()

    a = torch.FloatTensor(np.random.random((real.size(0), 1, 1, 1, 1)))
    if torch.cuda.is_available():
        a = a.cuda()

    interp = (a*real + ((1-a)*fake)).requires_grad_(True)
    d_interp = discriminator(interp)
    fake_ = torch.cuda.FloatTensor(real.shape[0], 1).fill_(1.0).requires_grad_(False)
    gradients = torch.autograd.grad(
        outputs=d_interp, inputs=interp, grad_outputs=fake_,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) -1)**2).mean() * lambda_
    return gradient_penalty 

def preTrainD(Discriminator, Generator, train_sampler, optimizer_D, RA=True, iter_preTrainD=1e4, GP=True):
    print("********************preTrain Discriminator******************") 

    for iteration in list(range(1, int(iter_preTrainD)+1)): 
        in_pic = [] 
        label_pic = []

        in_pic, label_pic= next(train_sampler) 
        if RA:
            mask = get_textured_region(label_pic)
        in_pic = transformPET.normalize(in_pic.type(torch.FloatTensor).cuda())
        label_pic = transformPET.normalize(label_pic.type(torch.FloatTensor).cuda()) 
        mask = mask.type(torch.FloatTensor).cuda()
        # import pdb 
        # pdb.set_trace()
        #################
        #     train D
        #################
        Discriminator.train()
        Generator.eval()
        optimizer_D.zero_grad()
        out_img, loss_recon, loss_style = Generator(in_pic)  #fake image
        # print(gen_imgs.shape)
        if RA:
            ##Region Aware
            real = label_pic*mask
            fake = out_img*mask
        else:
            real = label_pic
            fake = out_img
        loss_fake = torch.mean(Discriminator(fake.data))
        loss_real =  torch.mean(Discriminator(real.data))
        Wessertein_D = loss_real-loss_fake
        loss_D = loss_fake-loss_real
        if GP:
            ##Gradient penalty
            loss_D +=GradientPenalty(Discriminator, real.data, fake.data, lambda_=10)
        loss_D.backward()
        optimizer_D.step() 

            
    return Discriminator, optimizer_D


def save_model(G_net_model,optimizer_G, save_dir, ex=""):
    save_path=os.path.join(save_dir, "Model")
    mkdir(save_path)
    G_save_path = os.path.join(save_path,'Generator{}.pth'.format(ex))
    torch.save(G_net_model.cpu().state_dict(), G_save_path)
    G_net_model.cuda()

    
    opt_G_save_path = os.path.join(save_path,'Optimizer_G{}.pth'.format(ex))
    torch.save(optimizer_G.state_dict(), opt_G_save_path)

def build_train_sampler(low_dose_list, data_root, batch_size, shuffle=True):
    data_length_info = {} 

    dataset = Train_Data(root_dir=data_root, mode="train", low_dose_list=low_dose_list)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle) 
    sampler = DataSampler(dataloader)

    
    print("data length: \n", dataset.length) 
    
    return sampler


set_seeds(seed=42)
res_num = 8

transformPET = transformPET(data_range=20.0, mode="linear")
evaluate_one = Metrics(metric = ['psnr', 'ssim'])
io=dataIO()

total_iteration = 3e5 
val_iteration = 1000
batch_size = 2
eps=1e-8

psnr_max=0

save_dir = "experiment/UniPET"
mkdir(os.path.join(save_dir, "Model"))
mkdir(os.path.join(save_dir, "Gimg"))
val_save_img_path=os.path.join(save_dir, "Gimg/val_img")
mkdir(val_save_img_path)

        
        
low_dose_list = ["15s", "30s", "60s", "90s"] 
data_root = "/home/data/zhiwen/dataset/"
train_sampler = build_train_sampler(low_dose_list, data_root, batch_size, shuffle=True)
valid_loader = DataLoader(Eval_Data(root_dir=data_root, mode="val", low_dose_list=low_dose_list), batch_size=1, shuffle=False) 



#### initiate G and D 

Generator = UniPET() 
Generator.cuda() 
Generator.load_state_dict(torch.load(os.path.join(save_dir, "Model","pretained_model.pth")))
Discriminator = Discriminator() 
Discriminator.cuda()


    
# import pdb 
# pdb.set_trace()

optimizer_G = torch.optim.AdamW(Generator.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-08)
optimizer_D = torch.optim.AdamW(Discriminator.parameters(), lr=1.0e-4, betas=(0.9, 0.99)) 


running_loss = []
eval_metrics={
    "psnr":[],
    "ssim":[],
    }

# 

Discriminator, optimizer_D = preTrainD(Discriminator, Generator, train_sampler, optimizer_D, RA=True, iter_preTrainD=1e4, GP=True)

pbar = tqdm(total=int(total_iteration))

print("################ Train ################")
for iteration in list(range(1, int(total_iteration)+1)):

    l_Condition=[]
    l_G=[]
    sinceEpoch=time.time()
 
    in_pic = [] 
    label_pic = []

    

    in_pic, label_pic = next(train_sampler) 
    mask = get_textured_region(label_pic)
    in_pic = transformPET.normalize(in_pic.type(torch.FloatTensor).cuda())
    label_pic = transformPET.normalize(label_pic.type(torch.FloatTensor).cuda()) 
    mask = mask.type(torch.FloatTensor).cuda()
    
    # import pdb 
    # pdb.set_trace()  
    
    #################
    #     train D
    #################
    Discriminator.train() 
    Generator.eval() 
    optimizer_D.zero_grad() 
    out_img, loss_recon, loss_style = Generator(in_pic, label_pic) 
    
    ### region-aware learning strategy
    real = label_pic*mask 
    fake = out_img*mask 
    loss_fake = torch.mean(Discriminator(fake.data)) 
    loss_real =  torch.mean(Discriminator(real.data)) 
    loss_D = loss_fake-loss_real 
    loss_D += GradientPenalty(Discriminator, real.data, fake.data, lambda_=10) 
    loss_D.backward() 
    optimizer_D.step()


    #################
    #     train G
    #################
    Generator.train()
    optimizer_G.zero_grad()
    out_img, loss_recon, loss_style = Generator(in_pic, label_pic) 

    fake = out_img*mask 
    loss_G = -torch.mean(Discriminator(fake)) 
    
    loss_total = loss_recon.mean() + 0.001*loss_style.mean() + 0.001*loss_G

    # print("after_inference_data:", get_current_memory_gb())


    loss_total.backward()
    optimizer_G.step()


    torch.cuda.empty_cache() 
            

    
    if iteration % val_iteration == 0: 
        psnr=0
        ssim=0
        Generator.eval() 
        for counter,data in enumerate(tqdm(valid_loader)):
            v_in_pic, v_label_pic, name, dose = data 
            # import pdb 
            # pdb.set_trace()
            v_in_pic = transformPET.normalize(v_in_pic.type(torch.FloatTensor).cuda()) 
            
            # import pdb 
            # pdb.set_trace()
            
            gen_img = merge_patch(Generator, v_in_pic)
            gen_img = transformPET.denormalize(gen_img) 
            
            v_label_pic = transformPET.clip(v_label_pic.type(torch.FloatTensor).cuda()) 
            
            oneEval = evaluate_one(gen_img,v_label_pic)
            psnr+=oneEval['psnr']
            ssim+=oneEval['ssim']
    
            io.save(gen_img.clone().detach().cpu().numpy().squeeze(), os.path.join(val_save_img_path, name[0].replace(".nii.gz", "_{}.nii.gz".format(dose[0]))))
            # import pdb 
            # pdb.set_trace()
            
            torch.cuda.empty_cache()
        c_psnr=psnr/(counter+1)
        c_ssim=ssim/(counter+1)
        eval_metrics['psnr'].append(c_psnr)
        eval_metrics['ssim'].append(c_ssim)
    
        save_model(Generator,optimizer_G,save_dir, "_running")
        if c_psnr>=psnr_max:
            psnr_max=c_psnr
            io.save("Best Iteration: {},\t PSNR: {},\t SSIM:{}, \t".format(iteration, c_psnr, c_ssim),os.path.join(save_dir, "best.txt"))
            save_model(Generator,optimizer_G, save_dir, "_best")
        io.save(
            {'eval_metrics':eval_metrics},
            os.path.join(save_dir, "evaluationLoss.bin")
            )

    pbar.set_description("loss_G:{:6}, psnr:{:6}".format(loss_G.item(), eval_metrics['psnr'][-1] if len(eval_metrics['psnr'])>0 else 0)) 
    pbar.update() 

