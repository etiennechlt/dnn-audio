import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
from torch import tensor
from scipy.io.wavfile import write
import os
from time import process_time

def save_bruit(filename, bruit, bruit_rs, RSB):
    waveform, sample_rate = torchaudio.load(filename)

# même temps, on choisi un morceau du bruit dans babble de manière random
    bruit_f = torch.narrow(bruit_rs,1,np.random.randint(0,(bruit.shape[1]-waveform.shape[1])/2),waveform.shape[1])

# on choisi le poid du bruit en fonction du RSB voulu
    alpha = sum(sum(abs(waveform)**2))/(tensor(float(RSB))*sum(sum(abs(bruit_f)**2)))
    s_br = waveform + alpha * bruit_f
    
    
# on calcul le spectrogramme
    spec_br, f_br, t_br, im_br = plt.specgram(s_br[0].t().numpy(), Fs = sample_rate, NFFT = 512, scale_by_freq = True, mode = 'psd', cmap = 'gray')
    
#on ajoute le spectro au dossier
    path_list = filename.split("/")

    
    br_name = 'b_' + path_list[3][:-3] + 'npy'
    
    np.save('../Data/sig/bruité/' + br_name, spec_br.astype(np.float32))

    print(br_name)
    """plt.axis('off')
    plt.savefig('../Data/sig/bruité/' + br_name,bbox_inches='tight')
    plt.clf()
    plt.cla()"""
    
    

def taille_sig(filepath):
    sig_tot = torch.tensor([])
    for subdir, dirs, files in os.walk(filepath):
        for file in files:
            if ('.WAV' in file):
                subpath = os.path.join(subdir, file)
                waveform, sample_rate = torchaudio.load(subpath)
                sig_tot = torch.cat((sig_tot,waveform),1)
    data = sig_tot[0].t().numpy()
    j=0
    i=0
    while (i < sig_tot.shape[1] - 48000) :
        sig = data[i:i+48000]
        path_list = filepath.split("/")
        sig_path = '/home/ptit/Documents/dnn-audio/Data/meme_taille_' + path_list[7][6:11] + '/sig_' + path_list[8][2] + '_' +str(j)+'.WAV' #il faut créer le dossier meme_taille_TRAIN ou meme_taille_TEST
        write(sig_path,sample_rate,sig)
        #torchaudio.save(sig_path,sig,sample_rate)
        j = j+1
        if (i > sig_tot.shape[1] - 48000):
            break
        else :
            i = i + 48000
