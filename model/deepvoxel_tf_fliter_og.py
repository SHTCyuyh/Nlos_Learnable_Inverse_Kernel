import torch
import torch.nn as nn
import numpy as np
from lib.customer_layers_3 import \
Transient2volumn, \
VisibleNet, \
Rendering

import gc
from method.tflct_og import tflct_og
from method.tflct_og_multikernel import tflct_og_multikernel
from method.tflct_low_multikernel import tflct_low_multikernel
from method.tfpf_og_multikernel import tfpf_og_multikernel
from method.tfpf_low_multikernel import tfpf_low_multikernel
from method.tfpf_low_multikernel2 import tfpf_low_multikernel2
from method.tfphasor import tfphasor
from method.tfphasor_lowfre import tfphasor_low
from method.tfphasor_atten import tfphasor_atten
from method.tfphasor_atten_lowpart import tfphasor_atten_low
from method.tfphasor_atten_fcachannel import tfphasor_atten_fcachannel
from method.tfphasor_atten_low_lowpart import tfphasor_atten_low_lowpart
from method.tfphasor_attenflatten import tfphasor_attenflatten
from method.tfpf_multikernel import tfpf_multikernel
from method.tfphasor_attenflatten_all import tfphasor_attenflatten_all
from method.tfphasor_atten_fusion import tfphasor_atten_fusion
from method.tfphasor_atten_h2hpart import tfphasor_atten_h2hpart
import pdb
###########################################################################
def normalize(data_bxcxdxhxw):
    b, c, d, h, w = data_bxcxdxhxw.shape
    data_bxcxk = data_bxcxdxhxw.reshape(b, c, -1)
    
    data_min = data_bxcxk.min(2, keepdim=True)[0]
    data_zmean = data_bxcxk - data_min
    
    # most are 0
    data_max = data_zmean.max(2, keepdim=True)[0]
    data_norm = data_zmean / (data_max + 1e-15)
    
    return data_norm.view(b, c, d, h, w)


################################################################
class DeepVoxels(nn.Module):

    def __init__(self,
                 nf0=16,
                 in_channels=3,
                 out_channels=3,
                 img_sidelength=256,
                 grid_dim=32,
                 time_full = 512,
                 time_re = 512,
                 bin_len=0.01,
                 wall_size=2.0,
                 mode='fk',
                 downflag= True,
                 res0=0,
                 init_flag = False):
        
        super(DeepVoxels, self).__init__()
        
        ###################################33    
        # pdb.set_trace()     
        imsz = img_sidelength
        sres = imsz // grid_dim
        tfull = time_full
        tres = sres
        
        ########################################################
        basedim = nf0
        self.basedim = basedim
        crop = tfull // tres
        bin_t = tres * (512 // tfull)
        # assert not raytracing
        flag = downflag
        self.downnet = Transient2volumn(nf0=basedim, in_channels=in_channels, stride=sres, flag=flag)
        self.initflag = init_flag
        
        print('bin_len %.7f' % bin_len *bin_t)

        method = self.get_method(mode)
        # pdb.set_trace()
        self.lct = method(spatial=imsz // sres, crop=tfull // tres, bin_len=bin_len * bin_t, \
                       wall_size=wall_size)
        
        layernum = 0
        self.visnet = VisibleNet(nf0=basedim * 1 + 1, layernum=layernum)
        
        self.depth = True
        assert out_channels == 6 or out_channels == 2
        pdb.set_trace()
        self.rendernet = Rendering(nf0=(basedim * 1 + 1) * (layernum // 2 * 2 + 1 + 1), out_channels=out_channels // 2,factor=sres)
        self.depnet = Rendering(nf0=(basedim * 1 + 1) * (layernum // 2 * 2 + 1 + 1), out_channels=out_channels // 2, isdep=True,factor=sres)
    
    def todev(self, dev):
        self.lct.todev(dev, self.basedim * 1 + 1)
    
    def get_method(self, mode):
        if mode == 'tfphasor': return tfphasor
        if mode == 'tfpf_low_multikernel2': return tfpf_low_multikernel2
        if mode == 'tfpf_og_multikernel': return tfpf_og_multikernel
        if mode == 'tflct_low_multikernel': return tflct_low_multikernel
        if mode == 'tflct_og_multikernel': return tflct_og_multikernel
        if mode == 'tflct_og': return tflct_og
        if mode == 'tfphasor_low': return tfphasor_low
        if mode == 'tfpf_low_multikernel': return tfpf_low_multikernel
        if mode == 'tfphasor_atten': return tfphasor_atten
        if mode == 'tfphasor_atten_low' : return tfphasor_atten_low
        if mode == 'tfphasor_atten_fcachannel': return tfphasor_atten_fcachannel
        if mode == 'tfphasor_atten_low_lowpart': return tfphasor_atten_low_lowpart
        if mode == 'tfphasor_attenflatten': return tfphasor_attenflatten
        if mode == 'tfpf_multikernel': return tfpf_multikernel  
        if mode == 'tfphasor_attenflatten_all': return tfphasor_attenflatten_all    
        if mode == 'tfphasor_atten_fusion': return tfphasor_atten_fusion 
        if mode == 'tfphasor_atten_h2hpart': return tfphasor_atten_h2hpart
    def noise(self, data):
        gau = 0.05 + 0.03 * torch.randn_like(data) + data
        poi = 0.03 * torch.randn_like(data) * gau + gau
        return poi

    def forward(self, input_voxel):
        
        if True:
            # pdb.set_trace()
            print(input_voxel.dtype)
            noisedata = self.noise(input_voxel)
        else:
            noisedata = input_voxel
        
        ###############################
        data_norm = normalize(noisedata)
        
        tfre = self.downnet(data_norm)  #4 32 128 32 32
        
        # lct
        # pdb.set_trace()
        tfre2 = self.lct(tfre)
        
        # resize
        x = tfre2
        zdim = x.shape[2]
        # zdimnew = zdim * 100 // 128
        zdimnew = zdim
        x = x[:, :, :zdimnew]
        tfre2 = x
        
        tfre2 = nn.ReLU()(tfre2)
        tfre2 = normalize(tfre2)
        
        ######################################
        # unet 2 voxel
        tfflat = self.visnet(tfre2)  # torch.Size([4, 16, 128, 64, 64]) -> torch.Size([4, 32, 64, 64])
        
        # render
        pdb.set_trace()
        rendered_img = self.rendernet(tfflat)
        
        if self.depth:
            dep_img = self.depnet(tfflat) ## torch.Size([4, 32, 64, 64]) -> torch.Size([4, 1, 256, 256])
            rendered_img = torch.cat([rendered_img, dep_img], dim=1)
        
        rendered_img = torch.clamp(rendered_img, 0, 1)
        rendered_img = rendered_img * 2 - 1
        return rendered_img



class Regularization(nn.Module):
    def __init__(self,model,weight_decay,p=2, paraname = ['lct.learnanle_w']):
        super(Regularization, self).__init__()
        self.model = model
        self.weight_decay = weight_decay
        self.p = p
        self.paraname = paraname
        self.weight_list = self.get_weight(model, paraname = self.paraname)
        self.weight_info(self.weight_list)

    def forward(self,model):
        self.weight_list = self.get_weight(model, paraname = self.paraname)
        reg_loss = self.regularization_loss(self.weight_list,self.weight_decay,p=self.p)
        return reg_loss


        
    def get_weight(self, model, paraname):
        weight_list = []
        for name, param in model.named_parameters():
            if 'lct.learnanle_w' in name:
                weight = (name, param)
                weight_list.append(weight)
            for p in paraname:
                if p in name :
                    weight = (name, param)
                    weight_list.append(weight)
        return weight_list

    def regularization_loss(self, weight_list, weight_decay, p=2):
        reg_loss = 0
        for name, w in weight_list:
            l2_reg = torch.norm(w, p=p)
            reg_loss = reg_loss + l2_reg
        reg_loss = weight_decay * reg_loss
        return reg_loss

    def weight_info(self, weight_list):
        print("---------------regularization weight---------------")
        for name, w in weight_list:
            print(name)
        print("---------------------------------------------------")
#################################################################
if __name__ == '__main__':
    

    import cv2
    from einops import rearrange
    in_channels = 1
    n_channels = 4
    # from config_tf_fliter import get_args
    # args = get_args()
    in_dim = 1
    out_dim = 1
    dim = 33
    grid = 128
    mode = 'tfpf_og_multikernel'
    model = DeepVoxels(img_sidelength=256,
        in_channels=in_dim,
        out_channels=out_dim * 2,
        nf0=dim,
        grid_dim=grid,
        mode=mode,
        # raytracing=args.raytracing > 0
        )
    path = '/data2/nlospose/pose_v2_noise/pose_00/train/meas/person00-00009.hdr'
    data = cv2.imread(path, -1)
    data = data / np.max(data)
    meas = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
    meas = meas / np.max(meas)
    meas = rearrange(meas, '(t h) w ->1 1 t h w', t=600)
    meas = meas[:, :, :512]
    x = torch.from_numpy(meas)
    x = x.to("cuda:2")
    model = model.to("cuda:2")
    model.todev('cuda:2')
    # x = torch.randn((2,4096,96))
    out = model(x)  #[B ,CLS]
    print(out.shape)

    
    