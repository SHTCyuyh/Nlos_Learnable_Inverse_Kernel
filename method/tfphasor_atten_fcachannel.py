

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

import  numpy as np
import sys
sys.path.append('/home/yuyh/new_nlos_fre/method')
from .helper import definePsf, resamplingOperator, \
waveconvparam, waveconv
import gc
sys.path.append('/home/yuyh/new_nlos_fre/')
# from lib.customer_layers_3 import \
# Transient2volumn, \
# VisibleNet, \
# Rendering, \
# Transient2volumn2, Transient2volumn3

class FCABlock(nn.Module):
    """
        FcaNet: Frequency Channel Attention Networks
        https://arxiv.org/pdf/2012.11879.pdf
    """
    def __init__(self, channel, reduction=16):
        super(FCABlock, self).__init__()
        mid_channel = channel // reduction
        # self.dct_weight = dct_weight
        dct_weight = np.ones(1, dtype=np.float32)
        tfdcweight = torch.from_numpy(dct_weight)
        self.dct_weight = nn.Parameter(tfdcweight)
        self.dct_weight.requires_grad = True
        self.excitation = nn.Sequential(
            nn.Linear(channel, mid_channel, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channel, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = torch.sum(x*self.dct_weight, dim=[2,3,4])
        z = self.excitation(y).view(b, c, 1, 1, 1)
        return x * z.expand_as(x)


class tfphasor_atten_fcachannel(nn.Module):
    
    def __init__(self, spatial=256, crop=512, \
                 bin_len=0.01, wall_size=2.0, \
                 sampling_coeff=2.0, \
                 cycles=5, in_chans=1,
                 f_channel = 32*2):
        super(tfphasor_atten_fcachannel, self).__init__()
        
        self.spatial_grid = spatial
        self.crop = crop
        assert 2 ** int(np.log2(crop)) == crop
        
        self.bin_len = bin_len
        self.wall_size = wall_size
        
        self.sampling_coeff = sampling_coeff
        self.cycles = cycles
        
        self.parpareparam()
        self.conv = nn.Conv3d(in_chans, in_chans, kernel_size=1, stride=1)
        self.softmax = nn.Softmax(dim=2)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.sigmoid = nn.Sigmoid()
        self.fac = FCABlock(channel=f_channel)
    
    #####################################################
    def parpareparam(self):
        
        self.c = 3e8
        self.width = self.wall_size / 2.0
        self.bin_resolution = self.bin_len / self.c
        self.trange = self.crop * self.c * self.bin_resolution
        
        ########################################################3
        temprol_grid = self.crop
        sptial_grid = self.spatial_grid
        
        wall_size = self.wall_size
        bin_resolution = self.bin_resolution
        
        sampling_coeff = self.sampling_coeff
        cycles = self.cycles
        
        ######################################################
        # Step 0: define virtual wavelet properties
        # s_lamda_limit = wall_size / (sptial_grid - 1);  # sample spacing on the wall
        # sampling_coeff = 2;  # scale the size of the virtual wavelength (usually 2, optionally 3 for noisy scenes)
        # virtual_wavelength = sampling_coeff * (s_lamda_limit * 2);  # virtual wavelength in units of cm
        # cycles = 5;  # number of wave cycles in the wavelet, typically 4-6
        
        s_lamda_limit = wall_size / (sptial_grid - 1);  # sample spacing on the wall
        virtual_wavelength = sampling_coeff * (s_lamda_limit * 2);  # virtual wavelength in units of cm
        self.virtual_wavelength = virtual_wavelength
        
        virtual_cos_wave_k, virtual_sin_wave_k = \
        waveconvparam(bin_resolution, virtual_wavelength, cycles)
        
        virtual_cos_sin_wave_2xk = np.stack([virtual_cos_wave_k, virtual_sin_wave_k], axis=0)
        
        # use pytorch conv to replace matlab conv
        self.virtual_cos_sin_wave_inv_2x1xk = torch.from_numpy(virtual_cos_sin_wave_2xk[:, ::-1].copy()).unsqueeze(1)
        
        ###################################################
        slope = self.width / self.trange
        psf = definePsf(sptial_grid, temprol_grid, slope)
        fpsf = np.fft.fftn(psf)
        # lct
        # invpsf = np.conjugate(fpsf) / (1 / self.snr + np.real(fpsf) ** 2 + np.imag(fpsf) ** 2)
        # bp
        invpsf = np.conjugate(fpsf)
        
        self.invpsf_real = torch.from_numpy(np.real(invpsf).astype(np.float32)).unsqueeze(0)
        self.invpsf_imag = torch.from_numpy(np.imag(invpsf).astype(np.float32)).unsqueeze(0)
        
        ######################################################
        mtx_MxM, mtxi_MxM = resamplingOperator(temprol_grid)
        self.mtx_MxM = torch.from_numpy(mtx_MxM.astype(np.float32))
        self.mtxi_MxM = torch.from_numpy(mtxi_MxM.astype(np.float32))
        
    def todev(self, dev, dnum):
        
        self.virtual_cos_sin_wave_inv_2x1xk_todev = self.virtual_cos_sin_wave_inv_2x1xk.to(dev)
        self.datapad_2Dx2Tx2Hx2W = torch.zeros((2 * dnum, 2 * self.crop, 2 * self.spatial_grid, 2 * self.spatial_grid), dtype=torch.float32, device=dev)
        
        self.mtx_MxM_todev = self.mtx_MxM.to(dev)
        self.mtxi_MxM_todev = self.mtxi_MxM.to(dev)
        
        self.invpsf_real_todev = self.invpsf_real.to(dev)
        self.invpsf_imag_todev = self.invpsf_imag.to(dev)

    def compute_re(self, datafre, bnum, dnum):
        b2d,t,h,w = datafre.shape
        datafre = datafre.view(bnum, 2*dnum,t, h, w)
        datafre_real = datafre.real
        datafre_imag = datafre.imag
        datafre_real = self.fac(datafre_real)
        datafre_imag = self.fac(datafre_imag)
        datafre_real = datafre_real.view(b2d,t,h,w)
        datafre_imag = datafre_imag.view(b2d,t,h,w)
        w1 = self.invpsf_real_todev
        w2 = self.invpsf_imag_todev
        re_real = datafre_real * w1 - datafre_imag * w2
        re_imag = datafre_real * w2 + datafre_imag * w1
        refre = torch.stack([re_real, re_imag], dim=4)
        refre = torch.view_as_complex(refre)
        re = torch.fft.ifftn(refre)
        return  re        


    def forward(self, feture_bxdxtxhxw):
        
        # 1 padd data with zero
        bnum, dnum, tnum, hnum, wnum = feture_bxdxtxhxw.shape
        # for tbe, ten in zip(tbes, tens):
        #     assert tbe >= 0
        #     assert ten <= self.crop
        dev = feture_bxdxtxhxw.device
        
        featpad_bxdxtxhxw = []
        for i in range(bnum):
            featpad_1xdxt1xhxw = torch.zeros((1, dnum, 0, hnum, wnum), dtype=torch.float32, device=dev)
            featpad_1xdxt2xhxw = torch.zeros((1, dnum, 0, hnum, wnum), dtype=torch.float32, device=dev)
            featpad_1xdxtxhxw = torch.cat([featpad_1xdxt1xhxw, feture_bxdxtxhxw[i:i + 1], featpad_1xdxt2xhxw], dim=2)
            featpad_bxdxtxhxw.append(featpad_1xdxtxhxw)
        featpad_bxdxtxhxw = torch.cat(featpad_bxdxtxhxw, dim=0)
        
        # 2 params
        assert hnum == wnum
        assert hnum == self.spatial_grid
        sptial_grid = hnum
        temprol_grid = self.crop
        tnum = self.crop
        
        ####################################################
        data_BDxTxHxW = featpad_bxdxtxhxw.view(bnum * dnum, tnum, hnum, wnum)
 
        data_BDxHxWxT = data_BDxTxHxW.permute(0, 2, 3, 1)
        data_BDHWx1xT = data_BDxHxWxT.reshape(-1, 1, tnum)
        knum = self.virtual_cos_sin_wave_inv_2x1xk.shape[2]
        phasor_data_cos_sin_BDHWx2x1T = F.conv1d(data_BDHWx1xT, self.virtual_cos_sin_wave_inv_2x1xk_todev, padding=knum//2)
        if knum % 2 == 0:
            data_BDHWx2xT = phasor_data_cos_sin_BDHWx2x1T[:, :, 1:]
        else:
            data_BDHWx2xT = phasor_data_cos_sin_BDHWx2x1T
        data_BDxHxWx2xT = data_BDHWx2xT.reshape(bnum * dnum, hnum, wnum, 2, tnum)
        data_2xBDxTxHxW = data_BDxHxWx2xT.permute(3, 0, 4, 1, 2)
        data_2BDxTxHxW = data_2xBDxTxHxW.reshape(2 * bnum * dnum, tnum, hnum, wnum)
        
        #############################################################    
        # Step 2: transform virtual wavefield into LCT domain
        # datapad_2BDx2Tx2Hx2W = torch.zeros((2 * bnum * dnum, 2 * temprol_grid, 2 * sptial_grid, 2 * sptial_grid), dtype=torch.float32, device=dev)
        datapad_2Dx2Tx2Hx2W = self.datapad_2Dx2Tx2Hx2W
        # create new variable
        datapad_B2Dx2Tx2Hx2W = datapad_2Dx2Tx2Hx2W.repeat(bnum, 1, 1, 1)
        # actually, because it is all zero so it is ok
        datapad_2BDx2Tx2Hx2W = datapad_B2Dx2Tx2Hx2W
        
        left = self.mtx_MxM_todev
        right = data_2BDxTxHxW.view(2 * bnum * dnum, temprol_grid, -1)
        tmp = torch.matmul(left, right)
        tmp2 = tmp.view(2 * bnum * dnum, temprol_grid, sptial_grid, sptial_grid)
        
        datapad_2BDx2Tx2Hx2W[:, :temprol_grid, :sptial_grid, :sptial_grid] = tmp2
        
        ###########################################################3
        # Step 3: convolve with backprojection kernel
        
        # datapad_BDx2Tx2Hx2Wx2 = torch.stack([datapad_BDx2Tx2Hx2W, torch.zeros_like(datapad_BDx2Tx2Hx2W)], dim=4)
        # datafre = torch.rfft(datapad_2BDx2Tx2Hx2W, 3, onesided=False)
        datafre = torch.fft.fftn(datapad_2BDx2Tx2Hx2W)
        re = self.compute_re(datafre, bnum, dnum)
        # debug = True
        # if debug:
        #      datafre_real = datafre.real
        #      datafre_imag = datafre.imag
             
        #      re_real = datafre_real * self.invpsf_real_todev - datafre_imag * self.invpsf_imag_todev
        #      re_imag = datafre_real * self.invpsf_imag_todev + datafre_imag * self.invpsf_real_todev
        #      refre = torch.stack([re_real, re_imag], dim=4)

        #      refre = torch.stack([re_real, re_imag], dim=4)
        #      refre = torch.view_as_complex(refre)
        #      re2 = torch.fft.ifftn(refre)

        
        volumn_2BDxTxHxWx2 = re[:, :temprol_grid, :sptial_grid, :sptial_grid]
        
        ########################################################################
        # Step 4: compute phasor field magnitude and inverse LCT
        
        cos_real = volumn_2BDxTxHxWx2[:bnum * dnum, :, :, :].real
        cos_imag = volumn_2BDxTxHxWx2[:bnum * dnum, :, :, :].imag
        
        sin_real = volumn_2BDxTxHxWx2[bnum * dnum:, :, :, :].real
        sin_imag = volumn_2BDxTxHxWx2[bnum * dnum:, :, :, :].imag
        
        sum_real = cos_real ** 2 - cos_imag ** 2 + sin_real ** 2 - sin_imag ** 2
        sum_image = 2 * cos_real * cos_imag + 2 * sin_real * sin_imag
        
        tmp = (torch.sqrt(sum_real ** 2 + sum_image ** 2) + sum_real) / 2
        # numerical issue
        tmp = F.relu(tmp, inplace=False)
        sqrt_sum_real = torch.sqrt(tmp)
        
        #####################################################################
        left = self.mtxi_MxM_todev
        right = sqrt_sum_real.view(bnum * dnum, temprol_grid, -1)
        tmp = torch.matmul(left, right)
        tmp2 = tmp.view(bnum * dnum, temprol_grid, sptial_grid, sptial_grid)
        volumn_BDxTxHxW = tmp2
        
        volumn_BxDxTxHxW = volumn_BDxTxHxW.view(bnum, dnum, self.crop, hnum, wnum)
        
        return volumn_BxDxTxHxW

def noise(data):
        gau = 0.05 + 0.03 * torch.randn_like(data) + data
        poi = 0.03 * torch.randn_like(data) * gau + gau
        return poi

def normalize(data_bxcxdxhxw):
    b, c, d, h, w = data_bxcxdxhxw.shape
    data_bxcxk = data_bxcxdxhxw.reshape(b, c, -1)
    
    data_min = data_bxcxk.min(2, keepdim=True)[0]
    data_zmean = data_bxcxk - data_min
    
    # most are 0
    data_max = data_zmean.max(2, keepdim=True)[0]
    data_norm = data_zmean / (data_max + 1e-15)
    
    return data_norm.view(b, c, d, h, w)

if __name__ == '__main__':
    import cv2
    from einops import rearrange
    test_phasor = True
    from skimage import transform
    # path = '/data2/nlospose/pose_v2_noise/pose_00/train/meas/person00-00009.hdr'
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    # datapath_test = '/data2/nlospose/chen_task/depthdataset2/test'
    # test_dataset = dataset(datapath_test, [128,128,256])
    # test_loader = DataLoader(test_dataset, batch_size=2,  shuffle=False, num_workers=16)
    # data = next(iter(test_loader))
    # data_bxcxdxhxw = data['data']
    path = '/home/yuyh/new_nlos_fre/test2561282.npy'
    data = np.load(path)
    data = torch.from_numpy(data)

    path2= '/data2/nlospose/chen_task/depthdataset2/data/train/meas/person02-00842.hdr'



    if test_phasor:
        model = tfphasor_atten_fcachannel(spatial= 32, crop=64, bin_len=0.01*8, \
                       wall_size=2)
        model.todev(dev='cpu',dnum=32)
        data2 = cv2.imread(path2, -1)
        data2 = data2 / np.max(data2)
        meas2 = cv2.cvtColor(data2, cv2.COLOR_BGR2GRAY)
        meas2 = meas2 / np.max(meas2)
        meas2 = rearrange(meas2, '(t h) w ->t h w', t=600)
        meas2 = meas2[:512, :, :]
        meas2_down = transform.resize(meas2, (256,128,128))
        # K = 1
        # for i in range(K):
        a = meas2
        a = (a[::2, :, :] + a[1::2, :, :]) / 2
        a = (a[:, ::2, :] + a[:, 1::2, :]) / 2
        a = (a[:, :, ::2] + a[:, :, 1::2]) / 2

        b = torch.from_numpy(meas2)
        b = rearrange(b, 't h w -> 1 1 t h w')
        b = F.interpolate(b, scale_factor=0.5)
        b = b.squeeze().numpy()

        # meas2 = rearrange(meas2, 't h w ->1 1 t h w')
        meas2_down = rearrange(meas2_down, 't h w ->1 1 t h w')
        x1 = torch.from_numpy(meas2_down)
        #1x2 = x2.astype(torch.float)
        # x1 = noise(x1)
        # x1 = normalize(x1)
        x1 = torch.zeros((1,32,64,32,32))
        out = model(x1)
        re = out.detach().numpy()[0,0]
        p = np.max(re, axis=0)
        p = p/np.max(p)
        cv2.imwrite(f'x1.png', p*255)
        a = rearrange(a, 't h w ->1 1 t h w')
        x2 = torch.from_numpy(a)
        #1x2 = x2.astype(torch.float)
        x2 = noise(x2)
        x2 = normalize(x2)
        out = model(x2)
        re = out.detach().numpy()[0,0]
        p = np.max(re, axis=0)
        p = p/np.max(p)
        cv2.imwrite(f'x2.png', p*255)
        b = rearrange(b, 't h w ->1 1 t h w')
        x3 = torch.from_numpy(b)
        #1x2 = x2.astype(torch.float)
        x3 = noise(x3)
        x3 = normalize(x3)
        out = model(x3)
        re = out.detach().numpy()[0,0]
        p = np.max(re, axis=0)
        p = p/np.max(p)
        cv2.imwrite(f'x3.png', p*255)

     

        # # x = noise(meas)
        # x = x2
        # print(torch.max(x))

        print('done')
