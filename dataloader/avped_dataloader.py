import os
import numpy as np
import numpy.random as rng
import matplotlib.pyplot as plt
import cv2
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from preprocess.audio_process import *
from preprocess.image_process import *

np.random.seed(42)

class AVpedLoader(Dataset):
    def __init__(self, annotation_lines,audio_path,image_path,detect_path,segmentation_path,gt_path,lidar_path,dark_aug=0,testing=0):
        super(AVpedLoader, self).__init__()
        self.annotation_lines   = annotation_lines
        self.audio_path         = audio_path
        self.image_path         = image_path
        self.gt_path            = gt_path
        self.dark_aug           = dark_aug
        self.detect_path        = detect_path
        self.segmentation_path  = segmentation_path
        self.lidar_path         = lidar_path
        self.testing            = testing


    def __len__(self):
        return len(self.annotation_lines)

    def __getitem__(self, index):

        name       = self.annotation_lines[index]
        audio_name  = os.path.join(self.audio_path,name[:-4]+'npy')
        image_name  = os.path.join(self.image_path,name[:-4]+'png')
        gt_name     = os.path.join(self.gt_path,name[:-4]+'npy')
        detect_name =  os.path.join(self.detect_path,name[:-4]+'npy')
        seg_name    = os.path.join(self.segmentation_path,name[:-4]+'npy')
        lidar_name  = os.path.join(self.lidar_path,name[:-4]+'bin')

        # audio   = np.load(audio_name[:])
        audio   = make_seq_audio(self.audio_path,name[:-4]+'npy')

        # spec = Audio2Spectrogram_npy(audio)
        # spec = np.transpose(spec,[2,0,1])
        # spec = torch.from_numpy(spec).float()

        audio   = np.transpose(audio,[1,0])
        spec       = Audio2Spectrogram(audio,sr=48000)
        spec       = spec.float()


        detect     = np.load(detect_name)

        image  = cv2.imread(image_name,cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image,(256,256))
        image,detect,brightness  = image_darkaug(image,detect,self.dark_aug)
        image         = np.transpose(image,[2,0,1])
        image         = torch.from_numpy(image).float()


        detect = detect[0]
        detect = torch.tensor(detect)

        gt      = np.load(gt_name)[0]
        gt     = torch.from_numpy(gt).float()

        segment = np.load(seg_name)
        segment = cv2.resize(segment,(256,256))
        segment     = torch.from_numpy(segment)
        segment = segment.float()


        if self.testing:
            lidar   = np.fromfile(lidar_name,dtype=np.float32,count=-1).reshape([-1,4])
            return spec,image,gt,detect,segment,lidar
        else:
            return spec,image,gt,detect,segment


