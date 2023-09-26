

from cv2 import meanShift
import torch
import torch.nn as nn
import torch.nn.functional as F

import  numpy as np
import sys
from helper import definePsf, resamplingOperator, \
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
        
        
        self.invpsf_real = torch.from_numpy(np.real(invpsf).astype(np.float32)).unsqueeze(0)
        self.invpsf_imag = torch.from_numpy(np.imag(invpsf).astype(np.float32)).unsqueeze(0)

        learnanle_w_re = torch.zeros_like(self.invpsf_real)
        learnanle_w_im = torch.zeros_like(self.invpsf_imag)
        learnanle_w = torch.cat([learnanle_w_re, learnanle_w_im], dim=0)
        learnanle_w.requires_grad = True
        self.learnanle_w = nn.Parameter(learnanle_w)
        
        ######################################################
        mtx_MxM, mtxi_MxM = resamplingOperator(temprol_grid)
        self.mtx_MxM = torch.from_numpy(mtx_MxM.astype(np.float32))
        self.mtxi_MxM = torch.from_numpy(mtxi_MxM.astype(np.float32))
        
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
        datafre = torch.rfft(datapad_BDx2Tx2Hx2W, 3, onesided=False)
        datafre_real = datafre[:, :, :, :, 0]
        datafre_imag = datafre[:, :, :, :, 1]
        w1 = self.learnanle_w_todev[0] + self.invpsf_real_todev
        w2 = self.learnanle_w_todev[1] + self.invpsf_imag_todev
        re_real = datafre_real * w1 - datafre_imag * w2
        re_imag = datafre_real * w2 + datafre_imag * w1
        refre = torch.stack([re_real, re_imag], dim=4)
        re = torch.ifft(refre, 3)
        
        volumn_BDxTxHxW = re[:, :temprol_grid, :sptial_grid, :sptial_grid, 0]
        
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
    import scipy.io as sio
    test_fk = True
    name = 'person00-01257'
    path = f'/storage/data/yuyh/depthdataset2/data/train/meas/{name}.hdr'
    # path = '/home/yuyh/NLOS_VIDEO/realdata_resize1031.mat'
    if test_fk:
        model = lct(spatial= 256, crop=512, bin_len=0.01, \
                     wall_size=2)

        data = cv2.imread(path, -1)
        data = data / np.max(data)
        meas = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
        meas = meas / np.max(meas)
        meas = rearrange(meas, '(t h) w ->t h w', t=600)
        meas = meas[:512, :, :]
        meas = torch.from_numpy(meas)

        print(torch.max(meas))
        meas = noise(meas)
        print(torch.max(meas))
        # meas = sio.loadmat(path)['meas']
        # meas = meas.astype(np.float32)
     
        # K = 3
        # for i in range(K):
        #      meas = (meas[::2, :, :] + meas[1::2, :, :]) / 2
        #      meas = (meas[:, ::2, :] + meas[:, 1::2, :]) / 2
        #      meas = (meas[:, :, ::2] + meas[:, :, 1::2]) / 2
        meas = rearrange(meas, 't h w ->1 1 t h w')

        # x = torch.from_numpy(meas)
        # x = x.astype(torch.float)
        meas = meas.to('cuda:0')
        model.todev(dev='cuda:0',dnum=1)
        out = model(meas)
        re = out.cpu().detach().numpy()[0,0]
        p = np.max(re, axis=0)
        p = p/np.max(p)
        cv2.imwrite(f'rebuttal/lct/{name}.png', p*255)
        print('done')
        print(out.shape)
    
    # import os
    # import cv2
    # import numpy as np
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
    # '''
    # fd = '/u6/a/wenzheng/remote2/code-nlos-git/OccludedSceneRep-2/code/pytorch-wz/dataloader_light22_bbox';
    # ims = []
    # tbe = -1
    # for i in range(512):
    #     name = '%s/2-%d.png' % (fd, i)
    #     if not os.path.isfile(name):
    #         ims.append(np.zeros((256, 256), dtype=np.uint8))
    #         continue
        
    #     if tbe < 0:
    #         tbe = i
        
    #     im = cv2.imread(name)
    #     imgt = im[:256, :256, :]
    #     im = im[:256, -256:, :]
    #     imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    #     ims.append(imgray)
    
    # rect_data_txhxw = np.array(ims, dtype=np.float32) / 255.0
    # rect_data_hxwxt = np.transpose(rect_data_txhxw, [1, 2, 0])
    # '''
    
    # from scipy.io import loadmat
    # torch.cuda.set_device(0)
    # # data = loadmat('/home/wenzheng/largestore/nlos-phasor/nlos-fk-master/statue.mat')
    # # rect_data_hxwxt = data['data']

    # from loaderdep2 import dataset 
    # from torch.utils.data import Dataset, DataLoader

    # datapath = '/data2/nlospose/chen_task/mini-dataset'
    # all_data = dataset(datapath)
    # train_size = int(len(all_data) * 0.8)
    # test_size = len(all_data) - train_size
    # train_dataset, test_dataset = torch.utils.data.random_split(all_data, [train_size, test_size])
    # train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=16)
    # test_loader = DataLoader(test_dataset, batch_size=1,  shuffle=True, num_workers=16)

    # # data  b_d_t_h_w

    # # measFile = r'/home/yuyh/NLOSFeatureEmbeddings/data/human_test/meas/00.hdr'
    # # meas = cv2.imread(measFile, -1)
    # # meas = meas / np.max(meas)
    # # meas = cv2.cvtColor(meas, cv2.COLOR_BGR2GRAY)
    # # meas = meas / np.max(meas)
    # # from einops import rearrange
    # # meas = rearrange(meas, '(t h) w ->h w t', t=600)
    # # print(meas.shape)
    # # rect_data_hxwxt = meas[:, :, :512]
    # for batch_idx, da in enumerate(train_loader):
    #     rect_data_bxdxtxhxw = da['data']
    #     crop = 512
    #     bin_len = 0.01
    #     K = 1
    #     name = da['name']
        
    #     sptial_grid = 256
    #     crop = 512
    #     bin_len = 0.01  # 0.01
        
    #     K = 1
    #     temds = False
    #     for k in range(K):
    #         # rect_data_bxdxtxhxw = rect_data_bxdxtxhxw[:,:,::2, :, :] + rect_data_bxdxtxhxw[:,:,1::2, :, :]
    #         # rect_data_bxdxtxhxw = rect_data_bxdxtxhxw[:,:,:, ::2, :] + rect_data_bxdxtxhxw[:,:,:, 1::2, :]
    #         # sptial_grid = sptial_grid // 2
    #         rect_data_bxdxtxhxw = rect_data_bxdxtxhxw[:,:,::2, :, :] + rect_data_bxdxtxhxw[:,:,1::2, :, :]
    #         crop = crop // 2
    #         bin_len = bin_len * 2 
            
        
        
    #     bnum = rect_data_bxdxtxhxw.shape[0]
    #     dnum = rect_data_bxdxtxhxw.shape[1]
    #     # rect_data_bxdxhxwxt = np.tile(rect_data_bxdxhxwxt, [bnum, dnum, 1, 1, 1])
    #     rect_data_bxdxtxhxw = rect_data_bxdxtxhxw.cuda()
        
        
    #     dev = 'cuda'
        
    #     #####################################################################
    #     lctlayer = lct(spatial=sptial_grid, crop=crop, bin_len=bin_len,
    #                    method='bp')
    #     lctlayer.todev(dev, dnum)
        
    #     tbe = 0 // (2 ** K)
    #     if temds:
    #         tlen = 512 // (2 ** K)
    #     else:
    #         tlen = 512// (2 ** K)
        
    #     for i in range(1):
            
    #         re = lctlayer(rect_data_bxdxtxhxw[:, :, :, :, tbe:tbe + tlen], \
    #                       [tbe, tbe], [tbe + tlen, tbe + tlen])
        
    #     volumn_MxNxN = re.detach().cpu().numpy()[0, -1]
        
    #     # get rid of bad points
    #     zdim = volumn_MxNxN.shape[0] * 100 // 128
    #     volumn_MxNxN = volumn_MxNxN[:zdim]
    #     print('volumn min, %f' % volumn_MxNxN.min())
    #     print('volumn max, %f' % volumn_MxNxN.max())
    #     # volumn_MxNxN[:5] = 0
    #     # volumn_MxNxN[-5:] = 0
        
    #     volumn_MxNxN[volumn_MxNxN < 0] = 0
    #     front_view = np.max(volumn_MxNxN, axis=0)
    #     svdir = '/home/yuyh/NLOSFeatureEmbeddings/lct_re'
    #     name1 = os.path.split(name[0])[1]
    #     n = name1.split('.',)[0]
    #     cv2.imwrite("{}/{}.png".format(svdir, n), 255*front_view / np.max(front_view))
    #     # cv2.imshow("gt", imgt)
    #     # cv2.waitKey()
        
    #     # volumn_ZxYxX = volumn_MxNxN
    #     # volumn_ZxYxX = volumn_ZxYxX / np.max(volumn_ZxYxX)
    # # for i, frame in enumerate(volumn_ZxYxX):
    # #     print(i)
    # #     cv2.imwrite("re1", frame)
    # #     cv2.imshimwriteow("re2", frame / np.max(frame))
    # #     # cv2.waitKey(0)
    
