from tkinter import image_types
from pip import main
import torch
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import numpy as np
from einops import rearrange
from scipy.io import savemat
from numpy.random import poisson
import torch.nn.functional as F
from skimage import transform
noise = False

class dataset(Dataset):
    def __init__(self, datapath, resolution=[256,256,512], noise= False):
        super().__init__()

        self.measFiles = []
        self.depFiles = []
        self.imFiles = []
        self.noise = noise
        # self.jointsFils = []
        self.depPath = os.path.join(datapath, 'depth')
        self.imPath = os.path.join(datapath, 'im')
        self.measPath = os.path.join(datapath, 'meas')
        measNames = os.listdir(self.measPath)
        self.resolution = resolution
        # volNames = os.listdir(self.volPath)
        for measName in measNames:
            assert os.path.splitext(measName)[1] == '.hdr', \
                f'Data type should be .hdr,not {measName} in {self.measPath}'
            measFile = os.path.join(self.measPath, measName)
            self.measFiles.append(measFile)

            depFile = os.path.join(self.depPath, os.path.splitext(measName)[0] + '.hdr')
            assert os.path.isfile(depFile), \
                f'Do not have related vol {depFile}'
            self.depFiles.append(depFile)

            imFile = os.path.join(self.imPath, os.path.splitext(measName)[0] + '.hdr')
            assert os.path.isfile(depFile), \
                f'Do not have related vol {imFile}'
            self.imFiles.append(imFile)


    # def addnoise_dataset(self, meas):
    #     meas = rearrange(meas, 'a b ->(a b)')
    #     noised_meas = cv2.GaussianBlur(meas\
    #             , ksize=(0, 0), sigmaX=10.61, borderType=cv2.BORDER_REPLICATE) # 25 / 2.355
    #     noised_meas = poisson(noised_meas)
    #     return noised_meas
    
    def addnoise(self, data):
        h,w,t = data.shape
        gau = 0.05 + 0.03 * np.random.randn(h,w,t) + data
        poi = 0.03 * np.random.randn(h,w,t) * gau + gau
        return poi

    def __getitem__(self, index):
        measFile = self.measFiles[index]
        depFile = self.depFiles[index]
        imFile = self.imFiles[index]
        target_resolution = self.resolution

        try:
            meas = cv2.imread(measFile, -1)
            # print(meas.shape)
            meas = meas / np.max(meas)
            meas = cv2.cvtColor(meas, cv2.COLOR_BGR2GRAY)
            meas = meas / np.max(meas)
            # print(meas.shape)
        except TypeError:
            measFile = self.measFiles[0]
        #     jointFile = self.jointsFils[0]

            meas = cv2.imread(measFile, -1)
            meas = meas / np.max(meas)
            meas = cv2.cvtColor(meas, cv2.COLOR_BGR2GRAY)
            meas = meas / np.max(meas)
            print(
                f'--------------------\nNo.{index} meas is TypeError. \n--------------------------\n')
        except:
            measFile = self.measFiles[0]
            meas = cv2.imread(measFile, -1)
            meas = meas / np.max(meas)
            meas = cv2.cvtColor(meas, cv2.COLOR_BGR2GRAY)
            meas = meas / np.max(meas)
            print(
                f'--------------------\nNo.{index} meas is wrong. \n--------------------------\n')

        # meas = loadmat(measFile)['data_new'].astype(np.float32)
        if self.noise:
            # meas = self.addnoise_dataset(meas)
            # meas = rearrange(meas, '(t h w) 1  -> h w t', t = 600, h=256)
            meas = rearrange(meas, '(t h) w  -> h w t', t = 600)
            meas = meas[:,:,:512]
            meas = self.addnoise(meas)
        else:
            meas = rearrange(meas, '(t h) w  -> h w t', t = 600)
            meas = meas[:,:,:512]
        h,w,t = target_resolution
        if meas.shape != (h,w,t):
            meas = transform.resize(meas, (h,w,t))
        dep = cv2.imread(depFile, 0)
        dep = cv2.cvtColor(dep, cv2.COLOR_BGR2GRAY)
        dep = dep.astype(np.float32) / 255.0
        if dep.shape != (h,w):
            dep = cv2.resize(dep,(h,w))
        im = cv2.imread(imFile, 0)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im = im / np.max(im)
        im = im.astype(np.float32)
        if im.shape != (h,w):
            im = cv2.resize(im,(h,w))

        im_new = np.expand_dims(im, axis=2)
        dep_new =np.expand_dims(dep, axis=2)
        data = rearrange(meas, 'h w t -> 1 t h w')
        imgt1 = np.concatenate((im_new,dep_new), axis=2)
        imgt2 = (imgt1 + 1e-8) ** 0.5
        name = measFile


        data = {'data':data, 'imgt1':imgt1,\
                 'imgt2':imgt2, 'name':name}
        return data


    def __len__(self):
        
        return len(self.measFiles)
    


if __name__ == '__main__':
    testDataLoader = True
    if testDataLoader:
        datapath = '/data2/nlospose/chen_task/debug'
        all_data = dataset(datapath, resolution=[128,128,256])
        trainloader = DataLoader(all_data, batch_size=3, shuffle=True, num_workers=4)

        for batch_idx, data in enumerate(trainloader):
            meas = data['data']
            imgt1 = data['imgt1']
            imgt2 = data['imgt2']

        # dataiter = iter(trainloader)
        # data = dataiter.next()
        # meas = data['data']
        # imgt1 = data['imgt1']
        # imgt2 = data['imgt2']
        
        # print(meas.shape,imgt1.shape, imgt2.shape,len(trainloader),len(train_data))
        










# def get_data_loaders(folders, imszs=[600, 256, 256, 3], \
#                                     confocal=confocal, \
#                                     framenum=frame, \
#                                     gtsz=gtsz, \
#                                     datanum=datanum, \
#                                     mode='train', \
#                                     bs=bs, numworkers=thread):
#     files = os.listdir(folders)
