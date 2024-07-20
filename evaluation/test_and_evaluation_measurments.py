import scipy.io as sio
import numpy as np
from pystoi import stoi
import torch
import sys
import scipy.io.wavfile as wav
sys.path.append("../")
from utils.metrics import *
from utils.qualityMeasures import *
import os
#%% functions
def ifft_shift_RTFs(RTFs,device,Nl_out,Nr_out,M,wlen):
    """
    Performs shift on the given RTFs (Relative Transfer Functions).

    Args:
    - RTFs (torch.Tensor): Relative Transfer Functions with shape (batch_size, num_microphons, len_of_RTF)
    - device (torch.device): The device on which the computation will be performed
    - Nl_out (int): Left output RTF length
    - Nr_out (int): Right output RTF length
    - M (int): Number of microphones
    - wlen (int): Length of the window

    Returns:
    - RTFs_shift (torch.Tensor): Shifted Relative Transfer Functions with shape (batch_size, wlen, M)
    """
    len_of_RTF_out=Nl_out+Nr_out
    RTFs_= torch.zeros((RTFs.shape[0],wlen, M)).to(device)
    RTFs_[:,:Nr_out, 0] = RTFs[:,0,len_of_RTF_out * 0:len_of_RTF_out * 0 +Nr_out]
    RTFs_[:,:Nr_out, 1] = RTFs[:,1,len_of_RTF_out * 0:len_of_RTF_out * 0 +Nr_out]
    RTFs_[:,:Nr_out, 3] = RTFs[:,2,len_of_RTF_out * 0:len_of_RTF_out * 0 +Nr_out]
    RTFs_[:,:Nr_out, 4] = RTFs[:,3,len_of_RTF_out * 0:len_of_RTF_out * 0 +Nr_out]
    RTFs_[:,wlen-Nl_out:, 0] = RTFs[:,0,len_of_RTF_out * 1-Nl_out:len_of_RTF_out * 1]
    RTFs_[:,wlen-Nl_out:, 1] = RTFs[:,1,len_of_RTF_out * 1-Nl_out:len_of_RTF_out * 1]
    RTFs_[:,wlen-Nl_out:, 3] = RTFs[:,2,len_of_RTF_out * 1-Nl_out:len_of_RTF_out * 1]
    RTFs_[:,wlen-Nl_out:, 4] = RTFs[:,3,len_of_RTF_out * 1-Nl_out:len_of_RTF_out * 1]
    RTFs_[:, 2] = 1
    return RTFs_
def test(model, params,paths, test_loader, device):
    '''
    This function test the model on the test set. 
    It calculates the metrics and saves the results.
    Args:
        model (torch.nn.Module): The model to be tested
        params (Params): The parameters of the model
        results_path (string): The path to save the results
        test_loader (DataLoader): The dataloader of the test set
        device (torch.device): The device on which the computation will be performed
        epoc (int): The current epoc
    outputs:
        preformace (dict): The preformace of the model on the test set
    '''
    win = torch.hamming_window(params.wlen).to(device)
    
    model.eval()
    for data in test_loader:
        with torch.no_grad():
            data=data.to(device)
            x, edge_index, _ = data.x, data.edge_index, data.batch
            output = model(x, edge_index)   
            output=output[data.mask]    
            input=x[data.mask]
            # reshape output and input to RTFs shape 
            RTFs_first_spk_net= torch.reshape(output,(data.num_graphs,params.M-1,int(eval(params.feature_size)))).to(device)
            RTFs_first_spk_noisy=torch.reshape(input,(data.num_graphs,params.M-1,int(eval(params.feature_size)))).to(device)
            # get data for metrics calculation 
            y=torch.reshape(data.noisy_data,((data.num_graphs,params.fs*params.length_records,5))).to(device)
            first_spk=torch.reshape(data.data_ref_first_spk_M,((data.num_graphs,params.fs*params.length_records,5))).to(device)
            noise=torch.reshape(data.data_noise_M,((data.num_graphs,params.fs*params.length_records,5))).to(device)
            SNR_in=torch.reshape(data.data_SNR_in,((data.num_graphs,1))).to(device)
            First_spk_ref=torch.stft(first_spk[:,:,params.ref_mic],params.NFFT,params.n_hop,params.wlen,win,center=False,return_complex=True)
            first_spk_ref=torch.istft(First_spk_ref,params.NFFT,params.n_hop,params.wlen,win)
            Y_ref=torch.stft(y[:,:,params.ref_mic],params.NFFT,params.n_hop,params.wlen,win,center=False,return_complex=True)  
            y_ref=torch.istft(Y_ref,params.NFFT,params.n_hop,params.wlen,win)
            noise_ref=torch.stft(noise[:,:,params.ref_mic],params.NFFT,params.n_hop,params.wlen,win,center=False,return_complex=True)  
            noise_ref=torch.istft(noise_ref,params.NFFT,params.n_hop,params.wlen,win)
            y=torch.cat((y,y),dim=0)
            first_spk=torch.cat((first_spk,first_spk),dim=0)
            noise=torch.cat((noise,noise),dim=0)
            #%% get RTFs - Relative Transfer Functions 
            RTFs_first_spk_noisy=ifft_shift_RTFs(RTFs_first_spk_noisy,device,params.Nl,params.Nr,params.M,params.wlen)
            RTFs_first_spk_net=ifft_shift_RTFs(RTFs_first_spk_net,device,params.Nl,params.Nr,params.M,params.wlen)
            RTFs_first_spk_noisy=torch.cat((RTFs_first_spk_noisy,RTFs_first_spk_noisy),dim=0)
            RTFs_first_spk_net=torch.cat((RTFs_first_spk_net,RTFs_first_spk_net),dim=0)
            #%% get MVDR output 
            H_noisy_first_spk = torch.fft.fft(RTFs_first_spk_noisy, dim=1)
            H_net_first_spk = torch.fft.fft(RTFs_first_spk_net, dim=1)
            MVDR_noisy_y_stft,MVDR_noisy_first_spk_stft,MVDR_noisy_n_stft=MVDR_RTFs_batchs(first_spk,y,noise,H_noisy_first_spk.squeeze(),params,device,first_spk.shape[0])
            y_hat_noisy_first_channel=torch.istft(MVDR_noisy_y_stft,params.NFFT,params.n_hop,params.wlen,win)[0]
            first_spk_hat_noisy_first_channel=torch.istft(MVDR_noisy_first_spk_stft,params.NFFT,params.n_hop,params.wlen,win)[0]
            noise_hat_noisy_first_channel=torch.istft(MVDR_noisy_n_stft,params.NFFT,params.n_hop,params.wlen,win)[0]
            MVDR_gcn_y_stft,MVDR_gcn_first_spk_stft,MVDR_gcn_n_stft=MVDR_RTFs_batchs(first_spk,y,noise,H_net_first_spk.squeeze(),params,device,first_spk.shape[0])
            y_hat_gcn_first_channel=torch.istft(MVDR_gcn_y_stft,params.NFFT,params.n_hop,params.wlen,win)[0]
            first_spk_hat_gcn_first_channel=torch.istft(MVDR_gcn_first_spk_stft,params.NFFT,params.n_hop,params.wlen,win)[0]
            noise_hat_gcn_first_channel=torch.istft(MVDR_gcn_n_stft,params.NFFT,params.n_hop,params.wlen,win)[0]

           
            #%% SNR calculation 
            SNR_out_noisy_first_spk=10*torch.log10(torch.mean(first_spk_hat_noisy_first_channel[params.both_tim_st*params.fs:params.both_tim_fn*params.fs]**2)/torch.mean((noise_hat_noisy_first_channel[params.both_tim_st*params.fs:params.both_tim_fn*params.fs])**2)) 
            SNR_out_gcn_first_spk=10*torch.log10(torch.mean(first_spk_hat_gcn_first_channel[params.both_tim_st*params.fs:params.both_tim_fn*params.fs]**2)/torch.mean((noise_hat_gcn_first_channel[params.both_tim_st*params.fs:params.both_tim_fn*params.fs])**2)) 
            print('SNR in: {:.2f}'.format(SNR_in.cpu().numpy()[0][0]))
            print('SNR out GEVD: {:.2f}'.format(SNR_out_noisy_first_spk.cpu().numpy()))
            print('SNR out peerRTF: {:.2f}'.format(SNR_out_gcn_first_spk.cpu().numpy()))

            #%% STOI and ESTOI calculation  
    
            STOI_out_noisy_first_spk=stoi(first_spk_ref[0,params.both_tim_st*params.fs:params.both_tim_fn*params.fs].cpu(),y_hat_noisy_first_channel[params.both_tim_st*params.fs:params.both_tim_fn*params.fs].cpu(),params.fs)
            ESTOI_out_noisy_first_spk=stoi(first_spk_ref[0,params.both_tim_st*params.fs:params.both_tim_fn*params.fs].cpu(),y_hat_noisy_first_channel[params.both_tim_st*params.fs:params.both_tim_fn*params.fs].cpu(),params.fs,extended=True)

            STOI_in_first_spk=stoi(first_spk_ref[0,params.both_tim_st*params.fs:params.both_tim_fn*params.fs].cpu(),y_ref[0,params.both_tim_st*params.fs:params.both_tim_fn*params.fs].cpu(),params.fs)
            ESTOI_in_first_spk=stoi(first_spk_ref[0,params.both_tim_st*params.fs:params.both_tim_fn*params.fs].cpu(),y_ref[0,params.both_tim_st*params.fs:params.both_tim_fn*params.fs].cpu(),params.fs,extended=True)

            STOI_out_gcn_first_spk=stoi(first_spk_ref[0,params.both_tim_st*params.fs:params.both_tim_fn*params.fs].cpu(),y_hat_gcn_first_channel[params.both_tim_st*params.fs:params.both_tim_fn*params.fs].cpu(),params.fs)
            ESTOI_out_gcn_first_spk=stoi(first_spk_ref[0,params.both_tim_st*params.fs:params.both_tim_fn*params.fs].cpu(),y_hat_gcn_first_channel[params.both_tim_st*params.fs:params.both_tim_fn*params.fs].cpu(),params.fs,extended=True)
        
            print('STOI in: {:.2f}'.format(100*STOI_in_first_spk))
            print('STOI out GEVD: {:.2f}'.format(100*STOI_out_noisy_first_spk))
            print('STOI out peerRTF: {:.2f}'.format(100*STOI_out_gcn_first_spk))
            print('ESTOI in: {:.2f}'.format(100*ESTOI_in_first_spk))
            print('ESTOI out GEVD: {:.2f}'.format(100*ESTOI_out_noisy_first_spk))
            print('ESTOI out peerRTF: {:.2f}'.format(100*ESTOI_out_gcn_first_spk))


            #%% save input and outputs in wav files
            y_hat_noisy_first_channel = y_hat_noisy_first_channel.cpu().numpy()
            y_ref = y_ref.cpu().numpy()
            y_hat_gcn_first_channel = y_hat_gcn_first_channel.cpu().numpy()
            first_spk_ref = first_spk_ref.cpu().numpy()
            y_hat_noisy_normalized = y_hat_noisy_first_channel / np.max(np.abs(y_hat_noisy_first_channel))
            y_ref_normalized = y_ref[0] / np.max(np.abs(y_ref[0]))
            y_hat_gcn_normalized = y_hat_gcn_first_channel / np.max(np.abs(y_hat_gcn_first_channel))
            speaker_ref_normalized = first_spk_ref[0] / np.max(np.abs(first_spk_ref[0]))
            dir=paths.save_wav_dir+'wavs_epoc_100'
            if os.path.exists(dir)==False:
                os.makedirs(dir)
            wav.write(dir+ '/GEVD.wav',params.fs, y_hat_noisy_normalized[params.both_tim_st*params.fs:params.both_tim_fn*params.fs])
            wav.write(dir+ '/noisy signal.wav', params.fs, y_ref_normalized[params.both_tim_st*params.fs:params.both_tim_fn*params.fs])            
            wav.write(dir+ '/peerRTF.wav', params.fs, y_hat_gcn_normalized[params.both_tim_st*params.fs:params.both_tim_fn*params.fs])
            wav.write(dir+ '/clean signal.wav', params.fs, speaker_ref_normalized[params.both_tim_st*params.fs:params.both_tim_fn*params.fs])
