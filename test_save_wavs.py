import os
import torch
import scipy.io.wavfile as wav
from utils.metrics import *
from utils.qualityMeasures import *

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
    RTFs_[:,0, 2] = 1
    return RTFs_
def test(model, params,results_path, test_loader, device,epoc,flags):
    '''
    This function test the model on the test set. 
    It saves the results in the results_path directory as wav files.
    Args:
        model (torch.nn.Module): The model to be tested
        params (Params): The parameters of the model
        results_path (string): The path to save the results
        test_loader (DataLoader): The dataloader of the test set
        device (torch.device): The device on which the computation will be performed
        epoc (int): The current epoc
        flags (Flags): The flags of the model
    outputs:
        None
    '''
    epoc_for_save=100
    win = torch.hamming_window(params.wlen).to(device)
    save_idx=0
    #%% test
    model.eval()
    for data in test_loader:
    
        with torch.no_grad():
            data=data.to(device)
            if data.edge_index==None:
                RTFs_first_spk_net = model(data.first_spk_noisy)
                RTFs_first_spk_noisy = data.first_spk_noisy
            else:     
                x, edge_index, _ = data.x, data.edge_index, data.batch
                output = model(x, edge_index)   
                output=output[data.mask]    
                input=x[data.mask]
                #%% get RTFs from the model output and the input 
                RTFs_first_spk_net= torch.reshape(output,(data.num_graphs,params.M-1,int(eval(params.feature_size)))).to(device)
                RTFs_first_spk_noisy=torch.reshape(input,(data.num_graphs,params.M-1,int(eval(params.feature_size)))).to(device)
            RTFs_first_spk_oracle = torch.reshape(data.first_spk_clean,(data.num_graphs,params.M-1,int(eval(params.feature_size)))).to(device)

            #%% get data for MVDR 
            y=torch.reshape(data.noisy_data,((data.num_graphs,params.fs*params.length_records,5))).to(device)
            first_spk=torch.reshape(data.data_ref_first_spk_M,((data.num_graphs,params.fs*params.length_records,5))).to(device)
            noise=torch.reshape(data.data_noise_M,((data.num_graphs,params.fs*params.length_records,5))).to(device)
            SNR_in=torch.reshape(data.data_SNR_in,((data.num_graphs,1))).to(device)
            Y_ref=torch.stft(y[:,:,params.ref_mic],params.NFFT,params.n_hop,params.wlen,win,center=False,return_complex=True)  
            y_ref=torch.istft(Y_ref,params.NFFT,params.n_hop,params.wlen,win).cpu().numpy()
            noise_ref=torch.stft(noise[:,:,params.ref_mic],params.NFFT,params.n_hop,params.wlen,win,center=False,return_complex=True)  
            noise_ref=torch.istft(noise_ref,params.NFFT,params.n_hop,params.wlen,win)
          
            #%% get RTFs in the time domain and shift them 

            RTFs_first_spk_oracle=ifft_shift_RTFs(RTFs_first_spk_oracle,device,params.Nl,params.Nr,params.M,params.wlen).squeeze()
            RTFs_first_spk_noisy=ifft_shift_RTFs(RTFs_first_spk_noisy,device,params.Nl,params.Nr,params.M,params.wlen).squeeze()
            RTFs_first_spk_net=ifft_shift_RTFs(RTFs_first_spk_net,device,params.Nl,params.Nr,params.M,params.wlen).squeeze()
            #%% get RTFs in the frequency domain 
            H_oracle_first_spk = torch.fft.fft(RTFs_first_spk_oracle, dim=1)
            H_noisy_first_spk = torch.fft.fft(RTFs_first_spk_noisy, dim=1)
            H_net_first_spk = torch.fft.fft(RTFs_first_spk_net, dim=1)
            #%% get MVDR output 
            if epoc==epoc_for_save and flags.save_oracle_and_noisy:
                MVDR_oracle_y_stft,MVDR_oracle_first_spk_stft,MVDR_oracle_n_stft=MVDR_RTFs_batchs(first_spk,y,noise,H_oracle_first_spk.squeeze(),params,device,first_spk.shape[0])
                y_hat_oracle=torch.istft(MVDR_oracle_y_stft,params.NFFT,params.n_hop,params.wlen,win).cpu().numpy()
                
                MVDR_noisy_y_stft,MVDR_noisy_first_spk_stft,MVDR_noisy_n_stft=MVDR_RTFs_batchs(first_spk,y,noise,H_noisy_first_spk.squeeze(),params,device,first_spk.shape[0])
                y_hat_noisy=torch.istft(MVDR_noisy_y_stft,params.NFFT,params.n_hop,params.wlen,win).cpu().numpy()

            
            MVDR_gcn_y_stft,MVDR_gcn_first_spk_stft,MVDR_gcn_n_stft=MVDR_RTFs_batchs(first_spk,y,noise,H_net_first_spk.squeeze(),params,device,first_spk.shape[0])
            y_hat_gcn=torch.istft(MVDR_gcn_y_stft,params.NFFT,params.n_hop,params.wlen,win).cpu().numpy()
            
            #%% save wav files                 
            for example in range(first_spk.shape[0]):
                if epoc==epoc_for_save and flags.save_oracle_and_noisy:
    
                    if not os.path.exists(results_path+"wavs_epoc_"+str(epoc)+"/wav_files_oracle/"):
                        os.makedirs(results_path+"wavs_epoc_"+str(epoc)+"/wav_files_oracle/")
                    if not os.path.exists(results_path+"wavs_epoc_"+str(epoc)+"/wav_files_noisy/"):
                        os.makedirs(results_path+"wavs_epoc_"+str(epoc)+"/wav_files_noisy/")
                    if not os.path.exists(results_path+"wavs_epoc_"+str(epoc)+"/wav_files_ref/"):
                        os.makedirs(results_path+"wavs_epoc_"+str(epoc)+"/wav_files_ref/")
                if not os.path.exists(results_path+"wavs_epoc_"+str(epoc)+"/wav_files_gcn/"):
                    os.makedirs(results_path+"wavs_epoc_"+str(epoc)+"/wav_files_gcn/")


                # Create directories based on SNR_in value
                if epoc==epoc_for_save and flags.save_oracle_and_noisy:
                    snr_dir_oracle = results_path + "wavs_epoc_"+str(epoc)+"/wav_files_oracle/" + str(int(SNR_in[example].cpu().numpy()[0]))
                    snr_dir_noisy = results_path + "wavs_epoc_"+str(epoc)+"/wav_files_noisy/" + str(int(SNR_in[example].cpu().numpy()[0]))
                    snr_dir_ref = results_path + "wavs_epoc_"+str(epoc)+"/wav_files_ref/" + str(int(SNR_in[example].cpu().numpy()[0]))
                    if not os.path.exists(snr_dir_oracle):
                        os.makedirs(snr_dir_oracle)
                    if not os.path.exists(snr_dir_noisy):
                        os.makedirs(snr_dir_noisy)
                    if not os.path.exists(snr_dir_ref):
                        os.makedirs(snr_dir_ref)
                snr_dir_gcn = results_path + "wavs_epoc_"+str(epoc)+"/wav_files_gcn/" + str(int(SNR_in[example].cpu().numpy()[0]))  
                if not os.path.exists(snr_dir_gcn):
                    os.makedirs(snr_dir_gcn)
                
                # Save the audio files
                # Normalize the audio data to the range [-1, 1]
                if flags.save_oracle_and_noisy: 
                    y_hat_oracle_normalized = y_hat_oracle / np.max(np.abs(y_hat_oracle))
                    y_hat_noisy_normalized = y_hat_noisy / np.max(np.abs(y_hat_noisy))
                    y_ref_normalized = y_ref / np.max(np.abs(y_ref))
                y_hat_gcn_normalized = y_hat_gcn / np.max(np.abs(y_hat_gcn))

                # Save the normalized audio data as WAV files
                if flags.save_oracle_and_noisy: 
                    wav.write(snr_dir_oracle + '/y_'+str(save_idx) + ".wav", params.fs, y_hat_oracle_normalized[example,params.both_tim_st*params.fs:params.both_tim_fn*params.fs])
                    wav.write(snr_dir_noisy + '/y_'+str(save_idx) + ".wav", params.fs, y_hat_noisy_normalized[example,params.both_tim_st*params.fs:params.both_tim_fn*params.fs])
                    wav.write(snr_dir_ref + '/y_'+str(save_idx) + ".wav", params.fs, y_ref_normalized[example,params.both_tim_st*params.fs:params.both_tim_fn*params.fs])
                wav.write(snr_dir_gcn + '/y_'+str(save_idx) + ".wav", params.fs, y_hat_gcn_normalized[example,params.both_tim_st*params.fs:params.both_tim_fn*params.fs])

                save_idx+=1