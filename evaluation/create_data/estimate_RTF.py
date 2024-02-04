                                             
import numpy as np
import scipy.io as sio
import os
import scipy.signal as ss
from mat4py import loadmat
from scipy.linalg import eigh
from scipy.linalg import fractional_matrix_power
from scipy.fft import fft, ifft
import math

def creat_Qvv_Qzz(Y_STFT_matrix,params,noise_frm_st=0,noise_frm_fn=0,first_frm_st=0,first_frm_fn=0):
    '''
    This function creats the noise and speech covariance matrices
    Input: 
        Y_STFT_matrix: STFT of the signal
        params: parameters  
    Output:
        Rzz: noise covariance matrix
        Rvv: speech covariance matrix
    '''
    Rzz=np.zeros([int(eval(params.NUP)),params.M,params.M],dtype=complex)
    Rvv=np.zeros([int(eval(params.NUP)),params.M,params.M],dtype=complex)
    noise_frm_st = int(math.ceil(0*params.fs/params.n_hop))
    noise_frm_fn = int(math.floor(2*params.fs/params.n_hop-1))
    first_frm_st = int(math.ceil(2*params.fs/params.n_hop))
    first_frm_fn = int(math.floor(4*params.fs/params.n_hop)) 
    for k in range(int(eval(params.NUP))):
        Rvv[k,:,:]=Y_STFT_matrix[k,noise_frm_st:noise_frm_fn,:].T@Y_STFT_matrix[k,noise_frm_st:noise_frm_fn,:].conj()/len(Y_STFT_matrix[k,noise_frm_st:noise_frm_fn,:])
        Rzz[k,:,:]=Y_STFT_matrix[k,first_frm_st:first_frm_fn,:].T@Y_STFT_matrix[k,first_frm_st:first_frm_fn,:].conj()/len(Y_STFT_matrix[k,noise_frm_st:noise_frm_fn,:])
    return Rzz,Rvv

def GEVD(Qzz,Qvv,params):    
    '''
    This function estimate the RTFs using GEVD algorithm 
    Input: 
        Qzz: noise covariance matrix
        Qvv: speech covariance matrix
        params: parameters  
    Output:
        g: RTFs in time domain 
    ''' 
    a_hat_GEVD=np.zeros([int(eval(params.NUP)),params.M],dtype=complex)  
    G_full=np.zeros([params.NFFT,params.M],dtype=complex)
    for k in range(int(eval(params.NUP))):
        D_,V_ = eigh(Qvv[k,:,:])
        idx=np.flip(np.argsort(D_))
        D=D_[idx]
        V=V_[:,idx]
        D_matrix = np.diag(D, k=0)
        Rv1_2 = V @ fractional_matrix_power(D_matrix,1/2) @V.conj().T
        invRv1_2 = V @ fractional_matrix_power(D_matrix,-1/2) @ V.conj().T # inverse noise matrix     
        # Covariance whitening      
        Ry = invRv1_2@Qzz@invRv1_2.conj().T
        L, U = eigh(Ry[k,:,:])        
        idx=np.argmax(L)       
        temp = Rv1_2 @ U[:,idx]
        a_hat_GEVD[k,:] = temp/temp[params.ref_mic]
        ## remove outliers
    for m in range(params.M):
        ind = np.squeeze(np.array(np.where(abs(a_hat_GEVD[:,m])>3*np.mean(abs(a_hat_GEVD[:,m])))))
        if ind.size==1:
            G_full[params.NFFT//2]=1
        else:
            real = (2*np.random.binomial(1,0.5,len(ind))-1)*np.mean(np.real(a_hat_GEVD[:,m]))
            imag = (2*np.random.binomial(1,0.5,len(ind))-1)*np.mean(np.imag(a_hat_GEVD[:,m]))
            a_hat_GEVD[ind,m]=real+1j*imag
    ## reconstruct the RTFs         
    G_full[:int(eval(params.NUP))]=a_hat_GEVD
    G_full[int(eval(params.NUP)):]=np.flip(a_hat_GEVD[1:int(eval(params.NUP))-1], axis=0).conj()
    G_full[params.NFFT//2]=1
    # inverse fft to get the RTFs in time domain
    g=ifft(G_full[:,:].T).T
    return g

def  estimate_RTFs(paths,params):
    data = loadmat(paths.test_data_path+'test_data.mat')  
    y=np.array(data['y'])        
    frame_count = 1 + (y.shape[0] - params.wlen ) // params.n_hop
    Y_STFT_matrix=np.zeros([int(eval(params.NUP)),frame_count,params.M],dtype=complex)

    for m in range(params.M):
        Y_STFT_matrix[:,:,m]=ss.stft(y[:,m],params.fs, np.hamming(params.wlen) , nperseg=params.wlen, noverlap=params.wlen-params.n_hop, nfft=params.NFFT,boundary=None,padded=False)[2] 
    Qzz,Qvv=creat_Qvv_Qzz(Y_STFT_matrix,params)
    g=GEVD(Qzz,Qvv,params)
    #%% cut the RTFs to Nl_in and Nr_in and save them 
    h_cut=np.zeros([params.Nl_in+params.Nr_in,params.M])
    h_cut[:params.Nr_in,:]=np.real(g[:params.Nr_in,:])
    h_cut[params.Nr_in:,:]=np.real(g[params.NFFT-params.Nl_in:params.NFFT,:])
    RTFs=h_cut[:,[0,1,3,4]] # concatenate the RTFs to vector 
    RTFs_to_net=RTFs.flatten(order='F')
    train_data={}
    train_data['RTFs_to_net']=RTFs_to_net
    
    if not os.path.exists(paths.test_RTFs_path):
        os.makedirs(paths.test_RTFs_path)
    sio.savemat(paths.test_RTFs_path + 'RTFs_to_net.mat',train_data)
