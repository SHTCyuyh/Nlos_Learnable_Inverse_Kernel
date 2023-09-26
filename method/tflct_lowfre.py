
from cv2 import meanShift
import torch
import torch.nn as nn
import torch.nn.functional as F

import  numpy as np
import sys
from .helper import definePsf, resamplingOperator, \
filterLaplacian

class lct(nn.Module):
    
    def __init__(self, spatial=256, crop=512, \
                 bin_len=0.01, wall_size=2.0, \
                 method='lct', material='diffuse'):
        super(lct, self).__init__()
        
        self.spatial_grid = spatial
        self.crop = crop
        assert 2 ** int(np.log2(crop)) == crop
        
        self.bin_len = bin_len
        self.wall_size = wall_size
        
        #############################################################
        self.method = method
        self.material = material
        
        self.parpareparam()
    
    #####################################################
    def parpareparam(self):
        
        self.c = 3e8
        self.width = self.wall_size / 2.0
        self.bin_resolution = self.bin_len / self.c
        self.trange = self.crop * self.c * self.bin_resolution
        
        # maybe learnable?
        self.snr = 1e-1
        
        ########################################################3
        temprol_grid = self.crop
        sptial_grid = self.spatial_grid
        
        # 0-1
        gridz_M = np.arange(temprol_grid, dtype=np.float32)
        gridz_M = gridz_M / (temprol_grid - 1)
        gridz_1xMx1x1 = gridz_M.reshape(1, -1, 1, 1)
        self.gridz_1xMx1x1 = torch.from_numpy(gridz_1xMx1x1.astype(np.float32))
        
        ###################################################
        slope = self.width / self.trange
        psf = definePsf(sptial_grid, temprol_grid, slope)
        fpsf = np.fft.fftn(psf)
        
        if self.method == 'lct':
            invpsf = np.conjugate(fpsf) / (1 / self.snr + np.real(fpsf) ** 2 + np.imag(fpsf) ** 2)
        elif self.method == 'bp':
            invpsf = np.conjugate(fpsf)
        
        T, H, W  = invpsf.shape
        self.invpsf_real = torch.from_numpy(np.real(invpsf).astype(np.float32)).unsqueeze(0)
        self.invpsf_imag = torch.from_numpy(np.imag(invpsf).astype(np.float32)).unsqueeze(0)
        
        learnanle_w_re = torch.zeros((T//2,H//2,W//2)).unsqueeze(0)
        learnanle_w_im = torch.zeros((T//2,H//2,W//2)).unsqueeze(0)
        learnanle_w = torch.cat([learnanle_w_re, learnanle_w_im], dim=0)
        learnanle_w.requires_grad = True
        self.learnanle_w = nn.Parameter(learnanle_w)
        
        ######################################################
        mtx_MxM, mtxi_MxM = resamplingOperator(temprol_grid)
        self.mtx_MxM = torch.from_numpy(mtx_MxM.astype(np.float32))
        self.mtxi_MxM = torch.from_numpy(mtxi_MxM.astype(np.float32))

        padW = (W - W//2) // 2
        padH = (H - H//2) // 2
        padT = (T - T//2) // 2

        self.pad = nn.ConstantPad3d((padW,padW,padH,padH,padT,padT),0) #l,r t,b f,b
        
        #############################################################
        if self.method == 'bp':
            lapw_kxkxk = filterLaplacian()
            k = lapw_kxkxk.shape[0]
            self.pad = nn.ReplicationPad3d(2)
            self.lapw = torch.from_numpy(lapw_kxkxk).reshape(1, 1, k, k, k)
        
    def todev(self, dev, dnum):
        self.gridz_1xMx1x1_todev = self.gridz_1xMx1x1.to(dev)
        self.datapad_Dx2Tx2Hx2W = torch.zeros((dnum, 2 * self.crop, 2 * self.spatial_grid, 2 * self.spatial_grid), dtype=torch.float32, device=dev)
        
        self.mtx_MxM_todev = self.mtx_MxM.to(dev)
        self.mtxi_MxM_todev = self.mtxi_MxM.to(dev)
        
        self.invpsf_real_todev = self.invpsf_real.to(dev)
        self.invpsf_imag_todev = self.invpsf_imag.to(dev)
        self.learnanle_w_todev = self.learnanle_w.to(dev)
        
        if self.method == 'bp':
            self.lapw_todev = self.lapw.to(dev)
    
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
        
        ####################################################
        # 3 run lct
        # assert bnum == 1
        data_BDxTxHxW = featpad_bxdxtxhxw.view(bnum * dnum, self.crop, hnum, wnum)
        
        gridz_1xMx1x1 = self.gridz_1xMx1x1_todev
        if self.material == 'diffuse':
            data_BDxTxHxW = data_BDxTxHxW * (gridz_1xMx1x1 ** 4)
        elif self.material == 'specular':
            data_BDxTxHxW = data_BDxTxHxW * (gridz_1xMx1x1 ** 2)
        
        ###############################################################
        # datapad_BDx2Tx2Hx2W = torch.zeros((bnum * dnum, 2 * temprol_grid, 2 * sptial_grid, 2 * sptial_grid), dtype=torch.float32, device=dev)
        datapad_Dx2Tx2Hx2W = self.datapad_Dx2Tx2Hx2W
        # datapad_Dx2Tx2Hx2W = torch.zeros((dnum, 2 * temprol_grid, 2 * sptial_grid, 2 * sptial_grid), dtype=torch.float32, device=dev)
        # create new variable
        datapad_BDx2Tx2Hx2W = datapad_Dx2Tx2Hx2W.repeat(bnum, 1, 1, 1)

        left = self.mtx_MxM_todev
        right = data_BDxTxHxW.view(bnum * dnum, temprol_grid, -1)
        tmp = torch.matmul(left, right)
        tmp2 = tmp.view(bnum * dnum, temprol_grid, sptial_grid, sptial_grid)
        
        datapad_BDx2Tx2Hx2W[:, :temprol_grid, :sptial_grid, :sptial_grid] = tmp2
        
        ####################################################################################
        # datapad_BDx2Tx2Hx2Wx2 = torch.stack([datapad_BDx2Tx2Hx2W, torch.zeros_like(datapad_BDx2Tx2Hx2W)], dim=4)
        datafre = torch.fft.fftn(datapad_BDx2Tx2Hx2W)
        datafre_real = datafre.real
        datafre_imag = datafre.imag
        # dsfre = torch.fft.fftshift(datafre_real)
        # dsfim = torch.fft.fftshift(datafre_imag)
        invsre = torch.fft.fftshift(self.invpsf_real_todev)
        invsim = torch.fft.fftshift(self.invpsf_imag_todev)
        w1_s = self.pad(self.learnanle_w_todev[0].unsqueeze(0)) + invsre
        w2_s = self.pad(self.learnanle_w_todev[1].unsqueeze(0)) + invsim
        w1 = torch.fft.ifftshift(w1_s)
        w2 = torch.fft.ifftshift(w2_s)
        re_real = datafre_real * w1 - datafre_imag * w2
        re_imag = datafre_real * w2 + datafre_imag * w1
        refre = torch.stack([re_real, re_imag], dim=4)
        refre = torch.view_as_complex(refre)
        re = torch.fft.ifftn(refre).real
        
        volumn_BDxTxHxW = re[:, :temprol_grid, :sptial_grid, :sptial_grid]
        
        #########################################################################
        left = self.mtxi_MxM_todev
        right = volumn_BDxTxHxW.reshape(bnum * dnum, temprol_grid, -1)
        tmp = torch.matmul(left, right)
        tmp2 = tmp.view(bnum * dnum, temprol_grid, sptial_grid, sptial_grid)
        
        # volumn_BDxTxHxW = F.relu(tmp2, inplace=False)
        volumn_BDxTxHxW = tmp2
        
        if self.method == 'bp':
            volumn_BDx1xTxHxW = volumn_BDxTxHxW.unsqueeze(1)
            lapw = self.lapw_todev
            volumn_BDx1xTxHxW = self.pad(volumn_BDx1xTxHxW)
            volumn_BDx1xTxHxW = F.conv3d(volumn_BDx1xTxHxW, lapw)
            volumn_BDxTxHxW = volumn_BDx1xTxHxW.squeeze(1)
            # seems border  is bad
            # if self.crop == 512:
            if True:
                volumn_BDxTxHxW[:, :1] = 0
                # volumn_BDxTxHxW[:, -10:] = 0
            # volumn_BDxTxHxW = F.relu(volumn_BDxTxHxW, inplace=False)
        
        volumn_BxDxTxHxW = volumn_BDxTxHxW.view(bnum, dnum, self.crop, hnum, wnum)
        
        return volumn_BxDxTxHxW
        
def noise(data):
        gau = 0.05 + 0.03 * torch.randn_like(data) + data
        poi = 0.03 * torch.randn_like(data) * gau + gau
        return poi

if __name__ == '__main__':
    import cv2
    from einops import rearrange
    test_lct = True
    path = '/data2/nlospose/pose_v2_noise/pose_00/train/meas/person00-00009.hdr'
    if test_lct:
        model = lct(spatial= 64, crop=128, bin_len=0.01 * 4, \
                       method='lct', wall_size=2)
        data = cv2.imread(path, -1)
        data = data / np.max(data)
        meas = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
        meas = meas / np.max(meas)
        meas = rearrange(meas, '(t h) w ->t h w', t=600)
        meas = meas[:512, :, :]
        K = 2
        for i in range(K):
             meas = meas[::2, :, :] + meas[1::2, :, :]
             meas = meas[:, ::2, :] + meas[:, 1::2, :]
             meas = meas[:, :, ::2] + meas[:, :, 1::2]
        meas = rearrange(meas, 't h w ->1 1 t h w')
        x = torch.from_numpy(meas)
        model.todev(dev='cpu',dnum=1)
        x = noise(x)
        out = model(x)
        print(out.shape)
    
