import os

import cv2
import numpy as np
import torch

from preprocess.audio_process import Audio2Spectrogram, make_seq_audio
from preprocess.image_process import image_darkaug, image_darkaug_test
from utils.metrics import calculate_map
from utils.visualization import plot_result


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def evaluate(model,val_anno,audio_path,image_path,lidar_path,gt_path,device,dark_aug,vis=0,save=0):
    # read data here
    gt_array = []
    predict_array = []

    for name in val_anno:
        audio_name  = os.path.join(audio_path,name[:-4]+'npy')
        image_name  = os.path.join(image_path,name[:-4]+'png')
        gt_name     = os.path.join(gt_path,name[:-4]+'npy')
        lidar_name  = os.path.join(lidar_path,name[:-4]+'bin')

        # audio   = np.load(audio_name[:])
        audio   = make_seq_audio(audio_path,name[:-4]+'npy')
        audio   = np.transpose(audio,[1,0])
        spec       = Audio2Spectrogram(audio,sr=48000)
        spec       = spec.float()

        image  = cv2.imread(image_name,cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image,(256,256))
        image,brightness  = image_darkaug_test(image,dark_aug)
        image         = np.transpose(image,[2,0,1])
        image         = torch.from_numpy(image).float()
        gt      = np.load(gt_name)[0]
        gt_array.append(gt)

        with torch.no_grad():

            spec, image = spec.to(device),image.to(device)
            p,d,s = model(spec.unsqueeze(0), image.unsqueeze(0))
            p = p.cpu().detach().numpy()[0]
            d = d.cpu().detach().numpy()[0]
            predict_array.append(p)
            print("------------")
            print(p)
            print(gt)
            print(softmax(d))

            #vis lidar with predicted bbox
            if vis:
                lidar   = np.fromfile(lidar_name,dtype=np.float32,count=-1).reshape([-1,4])
                plot_result(gt,p,lidar)

    gt_array = np.array(gt_array)
    predict_array = np.array(predict_array)
    print(gt_array.shape,predict_array.shape)
    #calculation
    Dx = np.mean(np.abs(gt_array[:,0] - predict_array[:,0]))
    Dy = np.mean(np.abs(gt_array[:,1] - predict_array[:,1]))
    print(Dx,Dy)
    map_3 = calculate_map(gt_array,predict_array,0.3)

    return Dx,Dy,map_3


