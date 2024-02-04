#%%
from scipy.stats import randint 
import glob
import soundfile as sf   
import os, random
import scipy.io as sio
import numpy as np
import scipy.signal as ss
import random 

def create_example(paths,params):
    #%% foldaers paths
   
    if not os.path.exists(paths.test_data_path):
        os.makedirs(paths.test_data_path)
    #%% load AIR matrixs
    AIR_matrix = sio.loadmat(paths.path_AIRs)
    AIR_matrix_Grid=AIR_matrix['AIRs']
    AIR_matrix_OOG = sio.loadmat(paths.path_AIRs_mirage_OOG)
    AIR_matrix_OOG=AIR_matrix_OOG['data']
    #%% Parameters 
    OOG_position=25
    #%% Create AIR of pink noise from OOG positions
    # Read noise file
    data, samplerate = sf.read(paths.path_pink_noise)                                            
    pink_noise = np.array(data) 
    # Resample data if needed
    number_of_samples = round(len(pink_noise) * float(params.fs) / samplerate)
    pink_noise = ss.resample(data, number_of_samples)
    noise_AIR_matrix=np.zeros([OOG_position,params.fs*10,AIR_matrix_OOG.shape[2]])
    for i in range(9,25):
        for m in range(AIR_matrix_OOG.shape[2]):
            noise_AIR_matrix[i,:,m]=np.convolve(pink_noise[:10*params.fs,i-9], AIR_matrix_OOG[i,:,m],'same')
    noise_lvls = np.array([-10, -6, -2, 2, 6, 10])
    y=np.zeros([10*params.fs,params.M])
    noise=np.zeros([10*params.fs,params.M])
    x_matrix=np.zeros([8*params.fs,params.M])
    x_pad=np.zeros([10*params.fs,params.M])
    SNR_num=randint.rvs(0,6)
    SNR=noise_lvls[SNR_num]
    index = randint.rvs(0, AIR_matrix_Grid.shape[0]-1)
    noise_points=random.sample(range(9, 25),1) # noise points
    # one direction pink noise
    n=np.squeeze(noise_AIR_matrix[noise_points,:,:])
    # white noise 
    w_n=np.random.randn(10*params.fs,5)
    Librti_path='/dsi/gannot-lab1/datasets/LibriSpeech/LibriSpeech/Test/'
    book_folder=random.choice(os.listdir(Librti_path))    
    while book_folder=='hist.png':
        book_folder=random.choice(os.listdir(Librti_path))    
    Librti_path=Librti_path+book_folder+'/'
    book_folder=random.choice(os.listdir(Librti_path))
    Librti_path=Librti_path+book_folder     
    # Extract only wav files
    wavs = []
    for filename in glob.glob(Librti_path+'/*.wav'):
        d,sampale_rate=sf.read(filename)
        wavs.append(d)   
    wav_num=randint.rvs(0,len(wavs))   
    s=wavs[wav_num] 
    while len(s)<8*params.fs:
        wav_num=randint.rvs(0,len(wavs))   
        s=np.concatenate((s,wavs[wav_num]))
    x=np.zeros([s.shape[0],params.M])
    for m in range(params.M):
        x[:,m]=np.convolve(s, AIR_matrix_Grid[index,:,m],'same')
        x_matrix[:,m]=x[:8*params.fs,m]
    G=sum(x_matrix[:,params.ref_mic]**2)/sum(n[2*params.fs:,params.ref_mic]**2)*10**(-SNR/10)
    G2=sum(x_matrix[:,params.ref_mic]**2)/sum(w_n[2*params.fs:,params.ref_mic]**2)*10**(-30/10)
    x_pad[2*params.fs:,:]=x_matrix
    noise=np.sqrt(G)*n+np.sqrt(G2)*w_n
    y=x_pad+noise
    train_data={}
    train_data['x']=x_pad
    train_data['y']=y
    train_data['n']=np.sqrt(G)*n
    train_data['SNR_in']=SNR
    print('SNR_in: ',SNR,' test index point: ',index)
    file_name=paths.test_data_path+"test_data.mat"
    sio.savemat(file_name, train_data)
