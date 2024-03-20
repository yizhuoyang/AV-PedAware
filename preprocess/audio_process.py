import os
from random import random

import cv2
import librosa
import matplotlib.pyplot as plt
import matplotlib
import torch


matplotlib.use('TKAgg')
from scipy import signal
import numpy as np
from scipy.signal import butter, lfilter
import torchaudio.transforms as T
import torchvision.transforms as trans

def normalization_processing(data):

    data_min = data.min()
    data_max = data.max()

    data = data - data_min
    data = data / (data_max-data_min)

    return data

def normalization_processing_torch(data):
    # Assuming data is a PyTorch tensor
    data_min = torch.min(data)
    data_max = torch.max(data)

    # Normalizing the data
    normalized_data = (data - data_min) / (data_max - data_min)

    return normalized_data

def normalization_processing_torch_all(data):
    for i in range(data.shape[0]):
        data[i,:] = normalization_processing_torch(data[i,:])
    return data


def Audio2Spectrogram(np_data,sr,normarlization=1,min_frequency=100,max_frequency=3000):

    np_data   = torch.tensor(np_data,dtype=torch.float32)

    melspectrogram = T.MelSpectrogram(
        sample_rate = sr,
        n_fft = 2048,
        hop_length=1024,
        n_mels=200,
        # f_min=min_frequency,
        # f_max=max_frequency,
        pad_mode='constant',
        norm='slaney',
        mel_scale='slaney',
        power=2,
    )
    spectrogram = melspectrogram(np_data)
    # print(spectrogram.shape)
    if normarlization!=0:
        # spectrogram = spectrogram/np.linalg.norm(spectrogram,axis=0,keepdims=True)
        spectrogram = normalization_processing_torch_all(spectrogram)
    resize_transform = trans.Resize((224,64),antialias=True)
    spectrogram = resize_transform(spectrogram)
    spectrogram = spectrogram[:,:64,:]
    # # resize_transform = trans.Resize((64,64),antialias=True)
    # spectrogram = resize_transform(spectrogram)

    return spectrogram

def make_seq_audio(audio_path,name):
    parts = name.split("/")
    index = parts[-1][:-4]
    past_audio = np.load(os.path.join(audio_path,name))
    for f in range(1,5):
        file_name   =  f"{parts[0]}/{int(index)-2*f}.npy"
        current_pos = np.load(os.path.join(audio_path,file_name))
        past_audio = np.concatenate((past_audio,current_pos),0)
    return past_audio


def Audio2Spectrogram_npy(np_data,crop=1,normarlization=1):
    spec_list = []

    for i in range(8):
        spectrogram = librosa.feature.melspectrogram(
            y=np_data[:,i],
            sr = 48000,
            n_fft = 2048,
            hop_length=1024,
            n_mels=200
        )
        if normarlization!=0:
            spectrogram = normalization_processing(spectrogram)
        if crop:
            spectrogram = cv2.resize(spectrogram,(64,512))
            spectrogram = spectrogram[:64,:]
        else:
            spectrogram= cv2.resize(spectrogram,(64,64))
        spec_list.append(spectrogram)
    spec_list = np.stack(spec_list,axis=-1)
    return spec_list
