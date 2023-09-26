

from cv2 import meanShift
import torch
import torch.nn as nn
import torch.nn.functional as F

import  numpy as np
import sys
from .helper import definePsf, resamplingOperator, \
filterLaplacian, waveconvparam

class tfpf_multikernel(nn.Module):
    
    def __init__(self, spatial=256, crop=512, \
                 bin_len=0.01, wall_size=2.0, \
                 sampling_coeff=5.0, \
                 cycles=4):
        super(tfpf_multikernel, self).__init__()
        
        self.spatial_grid = spatial
        self.crop = crop
        assert 2 ** int(np.log2(crop)) == crop
        
        self.bin_len = bin_len
        self.wall_size = wall_size
        
        self.sampling_coeff = sampling_coeff
        self.cycles = cycles
        
        self.parpareparam()
    
    #####################################################
    def parpareparam(self):
        
        self.c = 3e8
        self.width = self.wall_size / 2.0;
        self.bin_resolution = self.bin_len / self.c
        self.trange = self.crop * self.c * self.bin_resolution
        self.scale = 2
        self.number_kernel = self.scale ** 3
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
        T, H, W  = invpsf.shape       
        self.invpsf_real = torch.from_numpy(np.real(invpsf).astype(np.float32)).unsqueeze(0)
        self.invpsf_imag = torch.from_numpy(np.imag(invpsf).astype(np.float32)).unsqueeze(0)
        self.multi_invpsf_real = self.invpsf_real.repeat(self.number_kernel,1,1,1)
        self.multi_invpsf_imag = self.invpsf_imag.repeat(self.number_kernel,1,1,1)
        learnanle_w_re = torch.zeros(temprol_grid//self.scale, \
                                    sptial_grid//self.scale, sptial_grid//self.scale).unsqueeze(0)
        learnanle_w_im = torch.zeros(temprol_grid//self.scale, \
                                    sptial_grid//self.scale, sptial_grid//self.scale).unsqueeze(0)
        learnanle_w = torch.cat([learnanle_w_re, learnanle_w_im], dim=0)
        learnanle_w = learnanle_w[:,None,...]
        learnanle_w = learnanle_w.repeat(1,self.number_kernel,1,1,1)                           

        self.learnanle_w = nn.Parameter(learnanle_w)
        self.learnanle_w.requires_grad = True

        ######################################################
        mtx_MxM, mtxi_MxM = resamplingOperator(temprol_grid)
        self.mtx_MxM = torch.from_numpy(mtx_MxM.astype(np.float32))
        self.mtxi_MxM = torch.from_numpy(mtxi_MxM.astype(np.float32))
        
        padW = (W - sptial_grid//self.scale) // 2
        padH = (H - sptial_grid//self.scale) // 2
        padT = (T - temprol_grid//self.scale) // 2

        self.pad = nn.ConstantPad3d((padW,padW,padH,padH,padT,padT),0)
        
    def todev(self, dev, dnum):
        
        self.virtual_cos_sin_wave_inv_2x1xk_todev = self.virtual_cos_sin_wave_inv_2x1xk.to(dev)
        self.datapad_2Dx2Tx2Hx2W = torch.zeros((2 * dnum, 2 * self.crop, 2 * self.spatial_grid, 2 * self.spatial_grid), dtype=torch.float32, device=dev)
        
        self.mtx_MxM_todev = self.mtx_MxM.to(dev)
        self.mtxi_MxM_todev = self.mtxi_MxM.to(dev)
        
        self.multi_invpsf_real_todev = self.multi_invpsf_real.to(dev)
        self.multi_invpsf_imag_todev = self.multi_invpsf_imag.to(dev)
        self.learnanle_w_todev = self.learnanle_w.to(dev)

        
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
        datapad_2Dx2Tx2Hx2W = self.datapad_2Dx2Tx2Hx2W
        datapad_B2Dx2Tx2Hx2W = datapad_2Dx2Tx2Hx2W.repeat(bnum, 1, 1, 1)
        datapad_2BDx2Tx2Hx2W = datapad_B2Dx2Tx2Hx2W
        left = self.mtx_MxM_todev
        right = data_2BDxTxHxW.view(2 * bnum * dnum, temprol_grid, -1)
        tmp = torch.matmul(left, right)
        tmp2 = tmp.view(2 * bnum * dnum, temprol_grid, sptial_grid, sptial_grid)
        datapad_2BDx2Tx2Hx2W[:, :temprol_grid, :sptial_grid, :sptial_grid] = tmp2
        datafre = torch.fft.fftn(datapad_2BDx2Tx2Hx2W)
        datafre_real = datafre.real
        datafre_imag = datafre.imag
        invsre = torch.fft.fftshift(self.multi_invpsf_real_todev)
        invsim = torch.fft.fftshift(self.multi_invpsf_imag_todev)  
        results = torch.zeros_like(feture_bxdxtxhxw)
        results = results.reshape(-1, self.number_kernel, temprol_grid//self.scale, \
                                    sptial_grid//self.scale, sptial_grid//self.scale)
        for i in range(self.number_kernel): 
           w1_s = self.pad(self.learnanle_w_todev[0,i]) + invsre[i]
           w2_s = self.pad(self.learnanle_w_todev[1,i]) + invsim[i]
           w1 = torch.fft.ifftshift(w1_s)
           w2 = torch.fft.ifftshift(w2_s)
           re_real = datafre_real * w1 - datafre_imag * w2
           re_imag = datafre_real * w2 + datafre_imag * w1
           refre = torch.stack([re_real, re_imag], dim=4)
           refre = torch.view_as_complex(refre)
           re = torch.fft.ifftn(refre)
           volumn_2BDxTxHxW = re[:, :temprol_grid, :sptial_grid, :sptial_grid]
           cos_real = volumn_2BDxTxHxW[:bnum * dnum, :, :, :].real
           cos_imag = volumn_2BDxTxHxW[:bnum * dnum, :, :, :].imag
           
           sin_real = volumn_2BDxTxHxW[bnum * dnum:, :, :, :].real
           sin_imag = volumn_2BDxTxHxW[bnum * dnum:, :, :, :].imag
           
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
           volumn_BDxTxHxW = tmp.view(bnum * dnum, temprol_grid, sptial_grid, sptial_grid)
           volumn_BDxTxHxW = volumn_BDxTxHxW.reshape(-1, self.number_kernel, temprol_grid//self.scale, \
                                                    sptial_grid//self.scale, sptial_grid//self.scale)
           results[:,i] =  volumn_BDxTxHxW[:,i]

        results = results.view(bnum, dnum, self.crop, hnum, wnum)
        volumn_BxDxTxHxW = results
        
        return volumn_BxDxTxHxW



if __name__ == '__main__':
    import cv2
    from einops import rearrange
    test_phasor = True
    path = '/data2/nlospose/pose_v2_noise/pose_00/train/meas/person00-00009.hdr'
    if test_phasor:
        model = phasor(spatial= 128, crop=256, bin_len=0.01 * 2, \
                       wall_size=2)
        data = cv2.imread(path, -1)
        data = data / np.max(data)
        meas = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
        meas = meas / np.max(meas)
        meas = rearrange(meas, '(t h) w ->t h w', t=600)
        meas = meas[:512, :, :]
        K = 1
        for i in range(K):
             meas = meas[::2, :, :] + meas[1::2, :, :]
             meas = meas[:, ::2, :] + meas[:, 1::2, :]
             meas = meas[:, :, ::2] + meas[:, :, 1::2]
        meas = rearrange(meas, 't h w ->1 1 t h w')
        x = torch.from_numpy(meas)
        model.todev(dev='cpu',dnum=1)
        out = model(x)
        print(out.shape) 

