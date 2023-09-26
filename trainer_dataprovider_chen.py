import torch
import torch.nn as nn
import wandb
import time
import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange, repeat
import sys
from torch import optim
from tqdm import tqdm
# from dataset2 import dataset
# from loaderdep2 import dataset
from loaderdep import get_data_loaders
import cv2
# from models.loss import mpjpe, n_mpjpe, p_mpjpe
from torch.utils.data import DataLoader
import logging
from torch.utils.tensorboard import SummaryWriter
import pickle
import os
from method.tfphasor import tfphasor
from skimage import transform
_ = torch.manual_seed(1234569527)
np.random.seed(123456)
def testdata(name,flag=True):
    model = tfphasor(spatial= 128, crop=256, bin_len=0.01*2, \
                       wall_size=2)
    model.todev(dev='cpu',dnum=1)
    if flag == True:
        data2 = cv2.imread(name[0], -1)
        data2 = data2 / np.max(data2)
        meas2 = cv2.cvtColor(data2, cv2.COLOR_BGR2GRAY)
        meas2 = meas2 / np.max(meas2)
        meas2 = rearrange(meas2, '(t h) w ->t h w', t=600)
        meas2 = meas2[:512, :, :]
        meas2 = transform.resize(meas2, (256,128,128))
        meas2 = rearrange(meas2, 't h w ->1 1 t h w')
        x2 = torch.from_numpy(meas2)
        out = model(x2)
        re = out.detach().numpy()[0,0]
        p = np.max(re, axis=0)
        p = p/np.max(p)
        cv2.imwrite(f'ogdatapf.png', p*255)
    if flag == False:
        data = name.detach().cpu()
        out = model(data)
        re = out.detach().numpy()[0,0]
        p = np.max(re, axis=0)
        p = p/np.max(p)
        cv2.imwrite(f'netdatapf.png', p*255)


def tv_loss(x, log_space=True, mode="l1", eps=0.1, reduction="sum"):
    if log_space:
        logx = torch.log(eps + x)
        diff = logx[..., 1:] - logx[..., :-1]
    else:
        diff = x[..., 1:] - x[..., :-1]

    if mode == "l1":
        loss = torch.sum(torch.abs(diff), -1)
    elif mode == "l2":
        loss = torch.sqrt(eps + torch.sum(diff ** 2, -1))
    else:
        raise NotImplementedError("invalid TV mode: {:s}".format(mode))
    if reduction == "mean":
        return torch.mean(loss)
    else:
        return torch.sum(loss)

def setup_logger(name, save_dir, distributed_rank):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # don't log results for the non-master process
    if distributed_rank > 0:
        return logger
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, "log.txt"), mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
def msemask(pre, gt, msk):
    if pre.shape[1] > 1:
        msk = msk.repeat(1, pre.shape[1], 1, 1)
    loss = (pre - gt) ** 2
    return (loss * msk).sum() / msk.sum()

def mesmask_per(pre,gt,msk):
    if pre.shape[1] > 1:
       msk = msk.repeat(1, pre.shape[1], 1, 1)
    loss = (pre - gt) ** 2
    return torch.sum((loss * msk), dim=(1,2,3)) / torch.sum(msk, dim=(1,2,3))
    


def PSNR(img1, img2):
    if img1.max() <= 1. and  img2.max() <= 1.:
        img1 = img1 * 255
        img2 = img2 * 255
    mse = torch.mean((img1 - img2) ** 2)
    return 20 * torch.log10(255.0 / torch.sqrt(mse))



def train(model, train_loader, criterion, optimizer, cfg, epoch):
    
    since = time.time()  
    model.train()
    device = cfg.DEVICE
    # device = 'cpu'
    for batch_idx, da in enumerate(train_loader):
        if da is None:
            continue
        
        datanp_bxcxdxhxw = da['data']
        imnp_bxhxwxc = da['imgt1']
        imnp_bxcxhxw = np.transpose(imnp_bxhxwxc, [0, 3, 1, 2])
        
        im2np_bxhxwxc = da['imgt2']
        im2np_bxcxhxw = np.transpose(im2np_bxhxwxc, [0, 3, 1, 2])
        
        # tbe = [int(d) // tres for d in da['tbe']]
        # ten = [int(d) // tres for d in da['ten']]
        
        #########################################################
        data_bxcxdxhxw = torch.from_numpy(datanp_bxcxdxhxw).float()
        im_bxcxhxw = torch.from_numpy(imnp_bxcxhxw).float()
        im2_bxcxhxw = torch.from_numpy(im2np_bxcxhxw).float()
        
        data_bxcxdxhxw, im_bxcxhxw, im2_bxcxhxw = \
        data_bxcxdxhxw.to(device), im_bxcxhxw.to(device), im2_bxcxhxw.to(device)
        
        imgt = im_bxcxhxw[:, :in_dim]
        depgt = im_bxcxhxw[:, in_dim:]
        imgt2 = im2_bxcxhxw
        
        maskgt = ((depgt[:, :1] > -1) & (imgt[:, :1] > -0.4)).float()
        
        ###############################################################
        # tt(1)
        optimizer.zero_grad()
        pdb.set_trace()
        output = model(data_bxcxdxhxw)
        outdim = output.shape[1] // 2
        
        impre = output[:, :outdim]
        deppre = output[:, outdim:]
        impre2 = ((impre + 1.0) / 2 + 1e-8) ** 0.5
        
        ###########################################################
        # print(batch_idx)
        # tt(2)
        lossim = criterion(impre, imgt) \
        + 0.3 * criterion(impre2, imgt2)
        
        lossdep = msemask(deppre, depgt, maskgt)  # mse(deppre, depgt)
        
        loss = lossim + lossdep
        
        # tt(3)
        loss.backward()
        # tt(4)
        optimizer.step()
        # tt(5)
        ##########################################################################
        if batch_idx % 1 == 0:
            print('Train Epoch: {} [{}/{} ({:.4f}%)]\tLoss: {:.4f} {:.4f}'.format(
                epoch, batch_idx * len(im_bxcxhxw), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), lossim, lossdep))

def eval(model, dataloader, criterion, cfg, epoch):
    model.eval()
    ims = []
    lossnp = 0
    lossdepnp = 0
    psnrim = 0
    mse = 0
    lossnum = 0
    device = cfg.DEVICE
    img_sidelength,_,time_re = cfg.INPUTRESOLUTION
    svdir = f'./results_bike/{cfg.PROJECT_NAME}'
    os.makedirs(svdir, exist_ok=True)
    depth_error = {}
    
    for batch_idx, da in enumerate(dataloader):
        if da is None:
            continue      
        data_bxcxdxhxw = da['data']
        imnp_bxhxwxc = da['imgt1']
        # names = da['name']
        im_bxcxhxw = np.transpose(imnp_bxhxwxc, [0, 3, 1, 2])
        imnp_bxcxhxw = im_bxcxhxw  #2 2 128 128  
        im2np_bxhxwxc = da['imgt2']
        im2_bxcxhxw = np.transpose(im2np_bxhxwxc, [0, 3, 1, 2])
        im2np_bxcxhxw = im2_bxcxhxw
        ########################################################      
        
        data_bxcxdxhxw = torch.from_numpy(data_bxcxdxhxw)
        im_bxcxhxw = torch.from_numpy(im_bxcxhxw)
        im2_bxcxhxw = torch.from_numpy(im2np_bxcxhxw)
        data_bxcxdxhxw, im_bxcxhxw, im2_bxcxhxw = \
        data_bxcxdxhxw.to(device), im_bxcxhxw.to(device), im2_bxcxhxw.to(device)     
        imgt = im_bxcxhxw[:, :1]
        depgt = im_bxcxhxw[:, 1:]
        imgt2 = im2_bxcxhxw   
        maskgt = ((depgt[:, :1] > -1) & (imgt[:, :1] > -0.4)).float()      
        ########################################################
        data_bxcxdxhxw = data_bxcxdxhxw.to(torch.float)
        output = model(data_bxcxdxhxw)  #2 1 512 256 256 
        outdim = output.shape[1] // 2   
        impre = output[:, :outdim]
        deppre = output[:, outdim:]
        impre2 = ((impre + 1.0) / 2 + 1e-8) ** 0.5    
        ###############################################################
        lossim = criterion(impre, imgt) \
        + 0.3 * criterion(impre2, imgt2)
        psnr = PSNR(imgt, impre)
        mse = criterion(impre, imgt)      
        # lossdep = msemask(deppre, depgt, maskgt)  # mse(deppre, depgt)
        lossdep_per = mesmask_per(deppre, depgt, maskgt)
        lossdep = lossdep_per.sum() / len(lossdep_per)
        # assert len(lossdep_per) == len(names),'wrong name'
        # for name,lossidx in zip(names, lossdep_per):
        #     key = name.split('/',)[-1].split('.',)[0]
        #     value = float(lossidx.detach().cpu().numpy())
        #     depth_error[key] = value
            # depth_error.update(name=value)

        writer.add_scalar("test_lossim",lossim,epoch)
        writer.add_scalar("test_lossdep",lossdep,epoch)
        writer.add_scalar("test_psnr",psnr,epoch)
        writer.add_scalar("test_mse",mse,epoch)
        
        print('epoch {} batch {} loss {} {} psnr {}'.format(epoch, batch_idx, lossim, lossdep, psnr))
        
        # imnp_bxcxhxw = imnp_bxcxhxw.numpy()
        data = np.concatenate([imnp_bxcxhxw, output.detach().cpu().numpy()], axis=3)
        data = np.concatenate([data[:, :1], data[:, 1:]], axis=2)
        ims.append(data)
        lossnp += lossim.item()
        lossdepnp += lossdep.item()
        lossnum += 1
        psnrim += psnr.item()
    
    ims = np.concatenate(ims, axis=0)
    datanum = len(ims)
    colnum = min(16, datanum)
    a = np.zeros([0, colnum * img_sidelength * 2, 1])
    for i in range(datanum // colnum):
        imslice = [ims[d:d + 1] for d in range(i * colnum, i * colnum + colnum)]  # ims[i * 4:i * 4 + 4]
        imslice = np.concatenate(imslice, axis=3)
        a = np.concatenate((a, np.transpose(imslice[0], [1, 2, 0])), axis=0)
    
    lossnp = lossnp / (lossnum + 1e-8)
    lossdepnp = lossdepnp / (lossnum + 1e-8)
    psnrim = psnrim / (lossnum + 1e-8)
    a = (a * 255)
    cv2.imwrite('%s/test-%d-%d-%.4f-%.4f-%.4f.png' % (svdir, epoch, batch_idx, lossnp, lossdepnp,psnrim), a)
    # os.makedirs(f'{svdir}/depth_error', exist_ok=True)
    # f_save = open(f'{svdir}/depth_error/{epoch}.pkl', 'wb')
    # pickle.dump(depth_error, f_save)
    # f_save.close()



if __name__ == '__main__':
    import argparse
    import pdb
    import yaml
    # from config.config_tfpf_low_multikernel_sameweight import _C as cfg
    from config.config import get_cfg_defaults
    parser = argparse.ArgumentParser()
    def check_file(path):
       if not os.path.isfile(path):
        raise ValueError('file does not exist: %s' % path)
    # parser.add_argument('-m', '--mode', help='mode', default='tfpf_og_multikernel')
    parser.add_argument('-c', '--config', help='mode', default='tfpf_atten64_fusion_1e-3')
    parser.add_argument('-vf', '--val_freq', type=int, default=5, help='validation frequency (20 epoch)')
    parser.add_argument('-reg', '--reg_list', default=[], help='regloss_list')
    parser.add_argument('-regweight', '--regweight', type=float, default=0.001, help='regweight')
    parser.add_argument('-lr', '--lr', type=float, default=2e-4, help='learningrate')
    args = parser.parse_args()
    in_channels = 1
    base_config = './config_bike'
    # from config_tf_fliter import get_args
    # args = get_args()
    # pdb.set_trace()
    configfile = os.path.join(base_config, f'{args.config}.yaml')
    check_file(configfile)
    cfg = get_cfg_defaults()
    cfg.merge_from_file(configfile)
    cfg.freeze()
    # print(cfg)
    global_step = 0
    log_dir = f'./results_bike/{cfg.PROJECT_NAME}/log'
    writer = SummaryWriter(log_dir=log_dir, max_queue=1)
    writer.add_text('OUT_PATH', log_dir,0)
    logger = setup_logger(f"{cfg.PROJECT_NAME}", log_dir, 0)
    logger.info("Running with config:\n{}".format(cfg))
    # from nlosformer_fouriour_filter2_lctweight import nlosformer
    # def get_model(cfg):
    # model = get_model(cfg)
    from model.deepvoxel_tf_fliter_og import DeepVoxels,Regularization
    in_dim = 1
    out_dim = 1
    dim = cfg.CHANNEL - 1
    dim = 3
    grid = cfg.RESOLUTION
    grid = 128
    mode = cfg.MODE
    img_sidelength,_,time_re = cfg.INPUTRESOLUTION
    device = cfg.DEVICE
    # device = 'cpu'
    model = DeepVoxels(img_sidelength=img_sidelength,
            time_full = time_re,
            in_channels=in_dim,
            out_channels=out_dim * 2,
            nf0=dim,
            grid_dim=grid,
            time_re=grid * 2,
            mode=mode,
            init_flag=cfg.init_flag
            # raytracing=args.raytracing > 0
            )
    # reg_loss = Regularization(model, weight_decay=args.regweight, p=2, paraname=args.reg_list).to(device)
    # model = nlosformer(out_dim=1, imagesise=256, input_size=(512,256,256),patch_size=(16,16,16),embed_dim=96,depths=4)
    pdb.set_trace()
    loss = nn.MSELoss() 
    if cfg.FREEZEDOWNNET == True:
        for name, para in model.named_parameters():
            if 'downnet' in name:
                para.requires_grad = False
                logger.info(f'freeze_para:{name}')
    if cfg.DOWNNETPATH is not None:
       downnet_weights = torch.load(cfg.DOWNNETPATH)
       keys_del = []
       for key, para in downnet_weights.items():
           if 'downnet' not in key:keys_del.append(key)
       for key in keys_del:del downnet_weights[key] 
       miss, unexpected = model.load_state_dict(downnet_weights, strict=False)  
    if cfg.LOADALLWEIGHT is not None:
        weights = torch.load(cfg.LOADALLWEIGHT)
        model.load_state_dict(weights, strict=False)
    paras = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(
             paras,
             lr=args.lr,
             betas=(0.9, 0.999)
         )
    model = model.to(device)
    model.todev(device)
    # model.todev2(device)
    folders = ['/storage/data/yuyh/bike/bike']
    imszs = [600, 256, 256, 3]
    bs = cfg.TRAIN.BATCH_SIZE
    print(bs)
    train_loader = get_data_loaders(folders, \
                                    confocal=1, \
                                    imszs=imszs, \
                                    timebe=0, timeen=6, \
                                    framenum=512, \
                                    gtsz=(256, 256), \
                                    datanum=1000, \
                                    mode='train', \
                                    bs=bs, numworkers=0)
    test_loader = get_data_loaders(folders, \
                                    confocal=1, \
                                    imszs=imszs, \
                                    timebe=0, timeen=6, \
                                    framenum=512, \
                                    gtsz=(256, 256), \
                                    datanum=1000, \
                                    mode='train_val', \
                                    bs=bs, numworkers=0)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, [2, 10, 20], 0.1, last_epoch = -1
    )
    if cfg.TRAIN.BEGIN_EPOCH != 0: 
        # model = Meas2Pose(cfg).to(cfg.DEVICE)
        # model = nlosformer(input_size=(512,128,128),patch_size=(8,8,8)).to(cfg.DEVICE)
        model.load_state_dict(torch.load(f"./results_bike/{cfg.PROJECT_NAME}/checkpoint/{cfg.TRAIN.BEGIN_EPOCH}.pth"))
        if cfg.WANDB:
            # wandb.log({"Train begin ": cfg.TRAIN.BEGIN_EPOCH}, commit=True)
            print(f"Train begin :{cfg.TRAIN.BEGIN_EPOCH} ")
    for epoch in tqdm(range(cfg.TRAIN.BEGIN_EPOCH, cfg.TRAIN.END_EPOCH)):
        # train_data_prefetcher = data_prefetcher(train_dataloader, cfg.DEVICE)
        lr_scheduler.step()
        train(model, train_loader, loss, optimizer, cfg, epoch)
        if epoch % 5 == 0 or epoch == cfg.TRAIN.END_EPOCH-1:
            os.makedirs(f'./results_bike/{cfg.PROJECT_NAME}/checkpoint', exist_ok=True)
            torch.save(model.state_dict(), f"./results_bike/{cfg.PROJECT_NAME}/checkpoint/{epoch}.pth")
        if epoch % args.val_freq == 0 or epoch == cfg.TRAIN.END_EPOCH-1:
            with torch.no_grad():
                  eval(model, test_loader, loss, cfg, epoch)
    print('finish')