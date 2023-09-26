

from __future__ import print_function
from __future__ import division

import os
import glob
import pdb
import cv2
import scipy
import scipy.io as sio
import numpy as np
from scipy import ndimage

from torch.utils.data import Dataset, DataLoader
import time
from cv2 import imshow
from einops import rearrange
# in this file, we try to load video


#######################################################
class DataProvider(Dataset):
    """
    Class for the data provider
    """

    def __init__(self, datafolder, \
                 gtsz=(256, 256),\
                 mode='train', datadebug=False):
        
        self.mode = mode
        self.datadebug = datadebug  
        self.datafolder = datafolder
        self.datapath = os.path.join(datafolder,mode)
        self.gtsz = gtsz
        self.imnum = len(os.listdir(os.path.join(self.datapath,'meas')))
        ###########################################
        self.modeldirs = []  
        measfiles = glob.glob(os.path.join(self.datapath,'meas')+'/*')
        self.gtname = 'im'
        for meas in measfiles:
            self.modeldirs.append(meas)

        self.gray = True
        
        #########################################################
        print('initilize done')

    def __len__(self):
        return self.imnum

    def __getitem__(self, idx):
        return self.prepare_instance(idx)
    
    
    def prepare_instance(self, idx):
        re = {}
        re['valid'] = True

        try:
            data_3xtxhxw, imgt, imgt2,  name,flag = self.loaddata(self.modeldirs[idx], idx)
            re['data'] = data_3xtxhxw
            # re['tbe'] = tbe
            # re['ten'] = ten
            re['imgt1'] = imgt
            re['imgt2'] = imgt2
            re['valid'] = flag
            re['name'] = name
        except:
            re['valid'] = False
            return re

        return re
    def loaddata(self, measfile, modeldiridx=-1):
        
        # light is [0 0 1] 
        #/data2/nlospose/chen_task/rearrange_humandata/data/train/meas/person00-00043.hdr
        # pdb.set_trace()
        name = measfile.split('/',)[-1].split('.',)[0]
        imgtname = measfile.replace('meas', 'im')
        depthgtname = measfile.replace('meas', 'depth')
        # do we also normalize gt?
        # im = cv2.imread(imgtname, 0)
        imgt = sio.loadmat(imgtname)['imgt']
        
            
        # imgt1 = im / np.max(im)
        # im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        # im = im / np.max(im)
        imgt1 = imgt.astype(np.float32)
        imgt1 = np.expand_dims(imgt1, axis=2)
        # if self.gtsz[0] < imgt1.shape[0]:
        #     ratio = int(np.ceil(imgt1.shape[0] / self.gtsz[0] / 2)) * 2 + 1
        #     imgt1 = cv2.blur(imgt1, (ratio, ratio))
        #     imgt1 = cv2.resize(imgt1, self.gtsz)
        
        # if self.gray:
        #     imgt1 = cv2.cvtColor(imgt1, cv2.COLOR_BGR2GRAY)
        #     imgt1 = imgt1 / np.max(imgt1)
        #     imgt1 = np.expand_dims(imgt1, axis=2)
        
        
    
        # imdep = cv2.imread(depthgtname, 0)
        # # print(dep.shape)
        # # dep = dep / np.max(dep)
        # imdep = cv2.cvtColor(imdep, cv2.COLOR_BGR2GRAY)
        # if self.gtsz[0] < imdep.shape[0]:
        #     ratio = int(np.ceil(imdep.shape[0] / self.gtsz[0] / 2)) * 2 + 1
        #     imdep = cv2.blur(imdep, (ratio, ratio))
        #     imdep = cv2.resize(imdep, self.gtsz)         
        # imdep = imdep.astype(np.float32) / 255.0
        # if np.max(imdep) > (254.0 / 255.0):
        #     print('bad depth!!!')
        #     return None, None, None, None, False
        # else:
        imdep = sio.loadmat(depthgtname)['depthgt']
        imdep = np.expand_dims(imdep, axis=2)
                
        ########################################################
        # trainsent
        # meas = cv2.imread(measfile, -1)
        # # print(meas.shape)
        # meas = meas / np.max(meas)
        # meas = cv2.cvtColor(meas, cv2.COLOR_BGR2GRAY)
        # meas = meas / np.max(meas)
        # meas = rearrange(meas, '(t h) w  -> 1 t h w', t = 600)
        # meas = meas[:,:512,:,:]
        # imgt11 = np.concatenate([imgt1, imdep], axis=2)
        # imgt2 = (imgt11 + 1e-8) ** 0.5
        meas = sio.loadmat(measfile)['meas']
        meas = rearrange(meas, 't h w  -> 1 t h w')
        imgt11 = np.concatenate([imgt1, imdep], axis=2)
        imgt2 = (imgt11 + 1e-8) ** 0.5

        
        return meas, imgt11, imgt2, name, True


def collate_fn(batch_list):

    collated = {}
    batch_list = [data for data in batch_list if data['valid']]
    if len(batch_list) == 0:
        return None

    # keys = batch_list[0].keys()
    keys = []
    for key in keys:
        val = [item[key] for item in batch_list]
        collated[key] = val

    viewnum = 1
    keys = ['imgt1', 'imgt2']
    for key in keys:
        val = [item[key] for item in batch_list]
        try:
            val = np.stack(val, axis=0)
        except:
            pass
        collated[key] = val
            
    keys = ['data']
    for key in keys:
        val = [item[key] for item in batch_list]
        try:
            val = np.stack(val, axis=0)
        except:
            pass
        collated[key] = val

    keys = ['name']
    for key in keys:
        val = [item[key] for item in batch_list]
        collated[key] = val
        
    return collated


def get_data_loaders(folders, \
                     gtsz=(256, 256), \
                     mode='train', bs=1, numworkers=0):

    print('Building dataloaders')

    dataset_train = DataProvider(
                                 datafolder=folders, \
                                 gtsz=gtsz, \
                                 mode=mode, datadebug=False
                                 )
    
    # always true
    shuffle = True
    if mode == 'test' or mode == 'testreal':
        shuffle = False

    train_loader = DataLoader(dataset_train, batch_size=bs, \
                              shuffle=shuffle, num_workers=numworkers, collate_fn=collate_fn)

    print('{} num {}'.format(mode,len(dataset_train)))
    print('{} iter'.format(mode,len(train_loader)))

    return train_loader


##############################################
if __name__ == '__main__':
    import scipy.io as sio
    
    folders = ['/u6/a/wenzheng/remote2/dataset-generated/shapenet/03790512-render2',
               '/u6/a/wenzheng/remote2/dataset-generated/shapenet/03790512-render3',
               '/u6/a/wenzheng/remote2/dataset-generated/shapenet/03790512-render4']
    
    # folders = ['/u6/a/wenzheng/remote3/dataset-generated/shapenet-render/mnist-shape_0.6_1.2-shift_0.4_0.4_-0.4_0.4']
    folders = '/storage/data/yuyh/depthdataset2/data'
    folders = '/storage/data/yuyh/depth_vary'
    

    
    # train_loader = get_data_loaders(folders, \
    #                                 gtsz=(256, 256), \
    #                                 mode='train', \
    #                                 bs=1, numworkers=0)
    test_loader = get_data_loaders(folders, \
                                    gtsz=(256, 256), \
                                    mode='test', \
                                    bs=1, numworkers=0)
    ## train  train_val
    ###############################################
    svdir_train = '/storage/data/yuyh/depth_vary/train'
    svdir_test = '/storage/data/yuyh/depth_vary/test'
    for i, data in enumerate(test_loader):
        # pdb.set_trace()
        if data is None:
            continue
        # for key in ['data', 'imgt1', 'imgt2']:
            # print('{}, {}, {}, {}'.format(i, key, data[key].shape, data[key].dtype))
        meas = data['data'].squeeze()
        imgt = data['imgt1'][:,:,:,0].squeeze()
        depthgt = data['imgt1'][:,:,:,1].squeeze()
        name = data['name'][0]
    #     sio.savemat(f'{svdir_train}/im/{name}.mat', {'imgt':imgt})
    #     sio.savemat(f'{svdir_train}/depth/{name}.mat', {'depthgt':depthgt})
    #     sio.savemat(f'{svdir_train}/meas/{name}.mat', {'meas':meas})
            
    # for i, data in enumerate(test_loader):
    #     # pdb.set_trace()
    #     if data is None:
    #         continue
    #     # for key in ['data', 'imgt1', 'imgt2']:
    #         # print('{}, {}, {}, {}'.format(i, key, data[key].shape, data[key].dtype))
    #     meas = data['data'].squeeze()
    #     imgt = data['imgt1'][:,:,:,0].squeeze()
    #     depthgt = data['imgt1'][:,:,:,1].squeeze()
    #     name = data['name'][0]
    #     sio.savemat(f'{svdir_test}/im/{name}.mat', {'imgt':imgt})
    #     sio.savemat(f'{svdir_test}/depth/{name}.mat', {'depthgt':depthgt})
    #     sio.savemat(f'{svdir_test}/meas/{name}.mat', {'meas':meas})
            
        
        '''
        for j, im in enumerate(data['imori']):
            im = (im * 255).astype(np.uint8)
            imwrite('%d-%d.png' % (i, j), im)
        '''
