import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image, transforms
import os


def generate_specgram(filepath):
    for subdir, dirs, files in os.walk(filepath):
        for file in files:
            if ('.WAV' in file):
                subpath = os.path.join(subdir, file)
                save_specgram(subpath)

                
                

def save_specgram(filename):
#     filename = "../Data/speech_TRAIN_1/TIMIT_TRAIN_1/DR1/MEDR0/SA2.WAV"
    waveform, sample_rate = torchaudio.load(filename)
#     waveform[:][0] = waveform[:][0] / sample_rate
    data = waveform[0].t().numpy()

    #            
    spectrum, freqs, t, im = plt.specgram(data, Fs=sample_rate, NFFT=512, scale_by_freq=True, mode='psd', cmap='gray')
    
    
    path_list = filename.split("/")
    path_list[2] = 'specgram' + path_list[2][6:]
    
    
    spec_name = path_list[4] + '_' + path_list[5] + '_'+ path_list[-1][:-3] + 'png'
    spec_path = ''
    for s in path_list[0:3]:
        spec_path = spec_path + s +'/'
        try:
            os.mkdir(spec_path)
            print(spec_path)
        except:
            pass

    # On enlève les axes pour sauvegarder la figure puis on la clean
    plt.axis('off')
    plt.savefig(spec_path+spec_name, bbox_inches='tight')
    plt.clf()
    plt.cla()
    

