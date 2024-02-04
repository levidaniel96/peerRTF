import scipy.io as sio
import numpy as np
from pystoi import stoi
import torch
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
    RTFs_[:,0, 2] = 1
    return RTFs_
def test(model, params,results_path, test_loader, device,epoc,flags):
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
        flags (Flags): The flags of the model
    outputs:
        preformace (dict): The preformace of the model on the test set
    '''
    i=0
    epoc_for_save=100
    win = torch.hamming_window(params.wlen).to(device)
    
    #%% define lists- for saving the results 
    SNR_in_list=[]
    SNR_out_oracle_first_spk_list=[]
    SNR_out_noisy_first_spk_list=[]
    SNR_out_gcn_first_spk_list=[]
    NPM_out_oracle_first_spk_list=[]
    NPM_out_noisy_first_spk_list=[]
    NPM_out_gcn_first_spk_list=[]
    SBF_out_oracle_first_spk_list=[]
    SBF_out_noisy_first_spk_list=[]
    SBF_out_gcn_first_spk_list=[]
    Blocking_out_gcn_first_spk_list=[]
    Blocking_out_oracle_first_spk_list=[]
    Blocking_out_noisy_first_spk_list=[]
    Blocking_with_n_out_gcn_first_spk_list=[]
    Blocking_with_n_out_oracle_first_spk_list=[]
    Blocking_with_n_out_noisy_first_spk_list=[]       
    STOI_in_first_spk_list=[]
    STOI_out_oracle_first_spk_list=[]
    STOI_out_noisy_first_spk_list=[]
    STOI_out_gcn_first_spk_list=[]
    ESTOI_in_first_spk_list=[]
    ESTOI_out_oracle_first_spk_list=[]
    ESTOI_out_noisy_first_spk_list=[]
    ESTOI_out_gcn_first_spk_list=[]
    pesq_in_first_spk_list=[]
    pesq_out_oracle_first_spk_list=[]
    pesq_out_noisy_first_spk_list=[]
    pesq_out_gcn_first_spk_list=[]
    si_sdr_in_first_spk_list=[]
    si_sdr_out_oracle_first_spk_list=[]
    si_sdr_out_noisy_first_spk_list=[]
    si_sdr_out_gcn_first_spk_list=[]
    Cbak_in_first_spk_list=[]
    Cbak_out_oracle_first_spk_list=[]
    Cbak_out_noisy_first_spk_list=[]
    Cbak_out_gcn_first_spk_list=[]
    Csig_in_first_spk_list=[]
    Csig_out_oracle_first_spk_list=[]
    Csig_out_noisy_first_spk_list=[]
    Csig_out_gcn_first_spk_list=[]
    Covl_in_first_spk_list=[]
    Covl_out_oracle_first_spk_list=[]
    Covl_out_noisy_first_spk_list=[]
    Covl_out_gcn_first_spk_list=[]
    preformace={}
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
                # reshape output and input to RTFs shape 
                RTFs_first_spk_net= torch.reshape(output,(data.num_graphs,params.M-1,int(eval(params.feature_size)))).to(device)
                RTFs_first_spk_noisy=torch.reshape(input,(data.num_graphs,params.M-1,int(eval(params.feature_size)))).to(device)
            RTFs_first_spk_oracle = torch.reshape(data.first_spk_clean,(data.num_graphs,params.M-1,int(eval(params.feature_size)))).to(device)
            # get data for metrics calculation 
            y=torch.reshape(data.noisy_data,((data.num_graphs,params.fs*params.length_records,5))).to(device)
            first_spk=torch.reshape(data.data_ref_first_spk_M,((data.num_graphs,params.fs*params.length_records,5))).to(device)
            mics_first=first_spk[:,:,[0,1,3,4]]
            ref_first=first_spk[:,:,2]
            noise=torch.reshape(data.data_noise_M,((data.num_graphs,params.fs*params.length_records,5))).to(device)
            mics_n=noise[:,:,[0,1,3,4]]
            ref_n=noise[:,:,2] 
            SNR_in=torch.reshape(data.data_SNR_in,((data.num_graphs,1))).to(device)
            First_spk_ref=torch.stft(first_spk[:,:,params.ref_mic],params.NFFT,params.n_hop,params.wlen,win,center=False,return_complex=True)
            first_spk_ref=torch.istft(First_spk_ref,params.NFFT,params.n_hop,params.wlen,win)
            Y_ref=torch.stft(y[:,:,params.ref_mic],params.NFFT,params.n_hop,params.wlen,win,center=False,return_complex=True)  
            y_ref=torch.istft(Y_ref,params.NFFT,params.n_hop,params.wlen,win)
            noise_ref=torch.stft(noise[:,:,params.ref_mic],params.NFFT,params.n_hop,params.wlen,win,center=False,return_complex=True)  
            noise_ref=torch.istft(noise_ref,params.NFFT,params.n_hop,params.wlen,win)
            #%% get RTFs - Relative Transfer Functions 
            RTFs_first_spk_oracle=ifft_shift_RTFs(RTFs_first_spk_oracle,device,params.Nl,params.Nr,params.M,params.wlen).squeeze()
            RTFs_first_spk_noisy=ifft_shift_RTFs(RTFs_first_spk_noisy,device,params.Nl,params.Nr,params.M,params.wlen).squeeze()
            RTFs_first_spk_net=ifft_shift_RTFs(RTFs_first_spk_net,device,params.Nl,params.Nr,params.M,params.wlen).squeeze()
            #%% get MVDR output 
            H_oracle_first_spk = torch.fft.fft(RTFs_first_spk_oracle, dim=1)
            H_noisy_first_spk = torch.fft.fft(RTFs_first_spk_noisy, dim=1)
            H_net_first_spk = torch.fft.fft(RTFs_first_spk_net, dim=1)
            if epoc==epoc_for_save and flags.save_oracle_and_noisy:
                MVDR_oracle_y_stft,MVDR_oracle_first_spk_stft,MVDR_oracle_n_stft=MVDR_RTFs_batchs(first_spk,y,noise,H_oracle_first_spk.squeeze(),params,device,first_spk.shape[0])
                y_hat_oracle_first_channel=torch.istft(MVDR_oracle_y_stft,params.NFFT,params.n_hop,params.wlen,win)
                first_spk_hat_oracle_first_channel=torch.istft(MVDR_oracle_first_spk_stft,params.NFFT,params.n_hop,params.wlen,win)
                noise_hat_oracle_first_channel=torch.istft(MVDR_oracle_n_stft,params.NFFT,params.n_hop,params.wlen,win)   
                
                MVDR_noisy_y_stft,MVDR_noisy_first_spk_stft,MVDR_noisy_n_stft=MVDR_RTFs_batchs(first_spk,y,noise,H_noisy_first_spk.squeeze(),params,device,first_spk.shape[0])
                y_hat_noisy_first_channel=torch.istft(MVDR_noisy_y_stft,params.NFFT,params.n_hop,params.wlen,win)
                first_spk_hat_noisy_first_channel=torch.istft(MVDR_noisy_first_spk_stft,params.NFFT,params.n_hop,params.wlen,win)
                noise_hat_noisy_first_channel=torch.istft(MVDR_noisy_n_stft,params.NFFT,params.n_hop,params.wlen,win)
            MVDR_gcn_y_stft,MVDR_gcn_first_spk_stft,MVDR_gcn_n_stft=MVDR_RTFs_batchs(first_spk,y,noise,H_net_first_spk.squeeze(),params,device,first_spk.shape[0])
            y_hat_gcn_first_channel=torch.istft(MVDR_gcn_y_stft,params.NFFT,params.n_hop,params.wlen,win)
            first_spk_hat_gcn_first_channel=torch.istft(MVDR_gcn_first_spk_stft,params.NFFT,params.n_hop,params.wlen,win)
            noise_hat_gcn_first_channel=torch.istft(MVDR_gcn_n_stft,params.NFFT,params.n_hop,params.wlen,win)  
            # save the results to mat files 
            if flags.save_mat and i<10:
                mat_file={}
                if flags.save_oracle_and_noisy and epoc==epoc_for_save:
                    mat_file['y_hat_oracle_first_channel']=y_hat_oracle_first_channel.cpu().numpy()
                    mat_file['first_spk_hat_oracle_first_channel']=first_spk_hat_oracle_first_channel.cpu().numpy()
                    mat_file['noise_hat_oracle_first_channel']=noise_hat_oracle_first_channel.cpu().numpy()
                    mat_file['y_ref']=y_ref.cpu().numpy()
                    mat_file['first_spk_ref']=first_spk_ref.cpu().numpy()
                    mat_file['noise_ref']=noise_ref.cpu().numpy()
                    mat_file['y_hat_noisy_first_channel']=y_hat_noisy_first_channel.cpu().numpy()
                    mat_file['first_spk_hat_noisy_first_channel']=first_spk_hat_noisy_first_channel.cpu().numpy()
                    mat_file['noise_hat_noisy_first_channel']=noise_hat_noisy_first_channel.cpu().numpy()
                    mat_file['y']=y.cpu().numpy()   
                    mat_file['first_spk']=first_spk.cpu().numpy()
                    mat_file['noise']=noise.cpu().numpy()
                mat_file['y_hat_gcn_first_channel']=y_hat_gcn_first_channel.cpu().numpy()
                mat_file['first_spk_hat_gcn_first_channel']=first_spk_hat_gcn_first_channel.cpu().numpy()
                mat_file['noise_hat_gcn_first_channel']=noise_hat_gcn_first_channel.cpu().numpy()
                mat_file['SNR_in']=SNR_in.cpu().numpy()
                if not os.path.exists(results_path+"/mat_files_epoc_"+str(epoc)):
                    os.makedirs(results_path+"/mat_files_epoc_"+str(epoc))
                sio.savemat(results_path+"/mat_files_epoc_"+str(epoc)+"/example_batch_"+str(i)+".mat", mat_file)
            #%% save RTFs to mat files 
            if flags.save_RTFs and i<10:
                RTFs={}
                if epoc==epoc_for_save and flags.save_oracle_and_noisy:
                    RTFs['RTFs_first_spk_oracle']=RTFs_first_spk_oracle.cpu().numpy()
                    RTFs['RTFs_first_spk_noisy']=RTFs_first_spk_noisy.cpu().numpy()
                RTFs['RTFs_first_spk_gcn']=RTFs_first_spk_net.cpu().numpy()
                if not os.path.exists(results_path+"/RTFs_epoc_"+str(epoc)):
                    os.makedirs(results_path+"/RTFs_epoc_"+str(epoc))
                sio.savemat(results_path+"/RTFs_epoc_"+str(epoc)+"/RTFs_batch_"+str(i)+".mat", RTFs)          
            #%% SNR calculation 
            if epoc==epoc_for_save and flags.save_oracle_and_noisy:
                SNR_out_oracle_first_spk=10*torch.log10(torch.mean(first_spk_hat_oracle_first_channel[:,params.both_tim_st*params.fs:params.both_tim_fn*params.fs]**2,dim=1)/torch.mean((noise_hat_oracle_first_channel[:,params.both_tim_st*params.fs:params.both_tim_fn*params.fs])**2,dim=1)) ######## 
                SNR_out_noisy_first_spk=10*torch.log10(torch.mean(first_spk_hat_noisy_first_channel[:,params.both_tim_st*params.fs:params.both_tim_fn*params.fs]**2,dim=1)/torch.mean((noise_hat_noisy_first_channel[:,params.both_tim_st*params.fs:params.both_tim_fn*params.fs])**2,dim=1)) ########
                for batch_idx in range (data.num_graphs):
                    SNR_out_oracle_first_spk_list.append(SNR_out_oracle_first_spk[batch_idx])
                    SNR_out_noisy_first_spk_list.append(SNR_out_noisy_first_spk[batch_idx])
        
            SNR_out_gcn_first_spk=10*torch.log10(torch.mean(first_spk_hat_gcn_first_channel[:,params.both_tim_st*params.fs:params.both_tim_fn*params.fs]**2,dim=1)/torch.mean((noise_hat_gcn_first_channel[:,params.both_tim_st*params.fs:params.both_tim_fn*params.fs])**2,dim=1)) ########
            for batch_idx in range (data.num_graphs):
                SNR_in_list.append(SNR_in[batch_idx])
                SNR_out_gcn_first_spk_list.append(SNR_out_gcn_first_spk[batch_idx])
            #%% NPM calculation
            if epoc==epoc_for_save and flags.save_oracle_and_noisy:
            
                NPM_out_oracle_first_spk=NPM(RTFs_first_spk_oracle,RTFs_first_spk_oracle)
                NPM_out_noisy_first_spk=NPM(RTFs_first_spk_noisy,RTFs_first_spk_oracle)
                for batch_idx in range (data.num_graphs):
                    NPM_out_oracle_first_spk_list.append(NPM_out_oracle_first_spk[batch_idx])
                    NPM_out_noisy_first_spk_list.append(NPM_out_noisy_first_spk[batch_idx])
            NPM_out_gcn_first_spk=  NPM(RTFs_first_spk_net,RTFs_first_spk_oracle)
            for batch_idx in range (data.num_graphs):     
                NPM_out_gcn_first_spk_list.append(NPM_out_gcn_first_spk[batch_idx])
            #%% SBF calculation
            if epoc==epoc_for_save and flags.save_oracle_and_noisy:
            
                SBF_out_oracle_first_spk=SBF(RTFs_first_spk_oracle,RTFs_first_spk_oracle,first_spk_ref)
                SBF_out_noisy_first_spk=SBF(RTFs_first_spk_noisy,RTFs_first_spk_oracle,first_spk_ref)
                for batch_idx in range (data.num_graphs):
                    SBF_out_oracle_first_spk_list.append(SBF_out_oracle_first_spk[batch_idx])
                    SBF_out_noisy_first_spk_list.append(SBF_out_noisy_first_spk[batch_idx])

            SBF_out_gcn_first_spk=  SBF(RTFs_first_spk_net,RTFs_first_spk_oracle,first_spk_ref)
            for batch_idx in range (data.num_graphs):
                SBF_out_gcn_first_spk_list.append(SBF_out_gcn_first_spk[batch_idx])
            if epoc==epoc_for_save and flags.save_oracle_and_noisy:
                Blocking_out_oracle_first_spk=Blocking(RTFs_first_spk_oracle, ref_first,mics_first)
                Blocking_out_noisy_first_spk=Blocking(RTFs_first_spk_noisy,ref_first,mics_first)
                for batch_idx in range (data.num_graphs):
                    Blocking_out_oracle_first_spk_list.append(Blocking_out_oracle_first_spk[batch_idx])
                    Blocking_out_noisy_first_spk_list.append(Blocking_out_noisy_first_spk[batch_idx])
            Blocking_out_gcn_first_spk=  Blocking(RTFs_first_spk_net,ref_first,mics_first)
            for batch_idx in range (data.num_graphs):
                Blocking_out_gcn_first_spk_list.append(Blocking_out_gcn_first_spk[batch_idx])
            if epoc==epoc_for_save and flags.save_oracle_and_noisy:
                
                Blocking_with_n_out_oracle_first_spk=Blocking_with_n(RTFs_first_spk_oracle,ref_first,mics_first,ref_n,mics_n)
                Blocking_with_n_out_noisy_first_spk=Blocking_with_n(RTFs_first_spk_noisy,ref_first,mics_first,ref_n,mics_n)
                for batch_idx in range (data.num_graphs):
                    Blocking_with_n_out_oracle_first_spk_list.append(Blocking_with_n_out_oracle_first_spk[batch_idx])
                    Blocking_with_n_out_noisy_first_spk_list.append(Blocking_with_n_out_noisy_first_spk[batch_idx])

            Blocking_with_n_out_gcn_first_spk=  Blocking_with_n(RTFs_first_spk_net,ref_first,mics_first,ref_n,mics_n)
            for batch_idx in range (data.num_graphs):
                Blocking_with_n_out_gcn_first_spk_list.append(Blocking_with_n_out_gcn_first_spk[batch_idx])
            #%% STOI and ESTOI calculation  
            if epoc==epoc_for_save and flags.save_oracle_and_noisy:
                for batch_idx in range(data.num_graphs):
    
                    STOI_out_oracle_first_spk=stoi(first_spk_ref[batch_idx,params.both_tim_st*params.fs:params.both_tim_fn*params.fs].cpu(),y_hat_oracle_first_channel[batch_idx,params.both_tim_st*params.fs:params.both_tim_fn*params.fs].cpu(),params.fs)
                    ESTOI_out_oracle_first_spk=stoi(first_spk_ref[batch_idx,params.both_tim_st*params.fs:params.both_tim_fn*params.fs].cpu(),y_hat_oracle_first_channel[batch_idx,params.both_tim_st*params.fs:params.both_tim_fn*params.fs].cpu(),params.fs,extended=True)
                    STOI_out_oracle_first_spk_list.append(STOI_out_oracle_first_spk)
                    ESTOI_out_oracle_first_spk_list.append(ESTOI_out_oracle_first_spk)

                    STOI_out_noisy_first_spk=stoi(first_spk_ref[batch_idx,params.both_tim_st*params.fs:params.both_tim_fn*params.fs].cpu(),y_hat_noisy_first_channel[batch_idx,params.both_tim_st*params.fs:params.both_tim_fn*params.fs].cpu(),params.fs)
                    ESTOI_out_noisy_first_spk=stoi(first_spk_ref[batch_idx,params.both_tim_st*params.fs:params.both_tim_fn*params.fs].cpu(),y_hat_noisy_first_channel[batch_idx,params.both_tim_st*params.fs:params.both_tim_fn*params.fs].cpu(),params.fs,extended=True)
                    STOI_out_noisy_first_spk_list.append(STOI_out_noisy_first_spk)
                    ESTOI_out_noisy_first_spk_list.append(ESTOI_out_noisy_first_spk)

            for batch_idx in range (data.num_graphs):
                STOI_in_first_spk=stoi(first_spk_ref[batch_idx,params.both_tim_st*params.fs:params.both_tim_fn*params.fs].cpu(),y_ref[batch_idx,params.both_tim_st*params.fs:params.both_tim_fn*params.fs].cpu(),params.fs)
                ESTOI_in_first_spk=stoi(first_spk_ref[batch_idx,params.both_tim_st*params.fs:params.both_tim_fn*params.fs].cpu(),y_ref[batch_idx,params.both_tim_st*params.fs:params.both_tim_fn*params.fs].cpu(),params.fs,extended=True)
                STOI_in_first_spk_list.append(STOI_in_first_spk)
                ESTOI_in_first_spk_list.append(ESTOI_in_first_spk)

                STOI_out_gcn_first_spk=stoi(first_spk_ref[batch_idx,params.both_tim_st*params.fs:params.both_tim_fn*params.fs].cpu(),y_hat_gcn_first_channel[batch_idx,params.both_tim_st*params.fs:params.both_tim_fn*params.fs].cpu(),params.fs)
                ESTOI_out_gcn_first_spk=stoi(first_spk_ref[batch_idx,params.both_tim_st*params.fs:params.both_tim_fn*params.fs].cpu(),y_hat_gcn_first_channel[batch_idx,params.both_tim_st*params.fs:params.both_tim_fn*params.fs].cpu(),params.fs,extended=True)
                STOI_out_gcn_first_spk_list.append(STOI_out_gcn_first_spk)
                ESTOI_out_gcn_first_spk_list.append(ESTOI_out_gcn_first_spk)
            #%% pesq calculation
            if epoc==epoc_for_save and flags.save_oracle_and_noisy:
                for batch_idx in range(data.num_graphs):          
                        _,pesq_out_oracle_first_spk=pesq(first_spk_ref[batch_idx,params.both_tim_st*params.fs:params.both_tim_fn*params.fs].cpu().numpy(),y_hat_oracle_first_channel[batch_idx,params.both_tim_st*params.fs:params.both_tim_fn*params.fs].cpu().numpy(),params.fs)
                        pesq_out_oracle_first_spk_list.append(pesq_out_oracle_first_spk)
                
                        _,pesq_out_noisy_first_spk=pesq(first_spk_ref[batch_idx,params.both_tim_st*params.fs:params.both_tim_fn*params.fs].cpu().numpy(),y_hat_noisy_first_channel[batch_idx,params.both_tim_st*params.fs:params.both_tim_fn*params.fs].cpu().numpy(),params.fs)
                        pesq_out_noisy_first_spk_list.append(pesq_out_noisy_first_spk)             
            for batch_idx in range(data.num_graphs):
                _,pesq_in_first_spk=pesq(first_spk_ref[batch_idx,params.both_tim_st*params.fs:params.both_tim_fn*params.fs].cpu().numpy(),y_ref[batch_idx,params.both_tim_st*params.fs:params.both_tim_fn*params.fs].cpu().numpy(),params.fs)
                pesq_in_first_spk_list.append(pesq_in_first_spk)

                _,pesq_out_gcn_first_spk=pesq(first_spk_ref[batch_idx,params.both_tim_st*params.fs:params.both_tim_fn*params.fs].cpu().numpy(),y_hat_gcn_first_channel[batch_idx,params.both_tim_st*params.fs:params.both_tim_fn*params.fs].cpu().numpy(),params.fs)
                pesq_out_gcn_first_spk_list.append(pesq_out_gcn_first_spk)
            #%% composite metrics calculation
            if epoc==epoc_for_save and flags.save_oracle_and_noisy:
                for batch_idx in range(data.num_graphs):
            
                        Csig_out_oracle_first_spk,Cbak_out_oracle_first_spk,Covl_out_oracle_first_spk=composite(first_spk_ref[batch_idx,params.both_tim_st*params.fs:params.both_tim_fn*params.fs].cpu().numpy(),y_hat_oracle_first_channel[batch_idx,params.both_tim_st*params.fs:params.both_tim_fn*params.fs].cpu().numpy(),params.fs)
                        Csig_out_oracle_first_spk_list.append(Csig_out_oracle_first_spk)
                        Cbak_out_oracle_first_spk_list.append(Cbak_out_oracle_first_spk)
                        Covl_out_oracle_first_spk_list.append(Covl_out_oracle_first_spk)
                        
                        Csig_out_noisy_first_spk,Cbak_out_noisy_first_spk,Covl_out_noisy_first_spk=composite(first_spk_ref[batch_idx,params.both_tim_st*params.fs:params.both_tim_fn*params.fs].cpu().numpy(),y_hat_noisy_first_channel[batch_idx,params.both_tim_st*params.fs:params.both_tim_fn*params.fs].cpu().numpy(),params.fs)
                        Csig_out_noisy_first_spk_list.append(Csig_out_noisy_first_spk)
                        Cbak_out_noisy_first_spk_list.append(Cbak_out_noisy_first_spk)
                        Covl_out_noisy_first_spk_list.append(Covl_out_noisy_first_spk)

            for batch_idx in range(data.num_graphs):
                Csig_in_first_spk,Cbak_in_first_spk,Covl_in_first_spk=composite(first_spk_ref[batch_idx,params.both_tim_st*params.fs:params.both_tim_fn*params.fs].cpu().numpy(),y_ref[batch_idx,params.both_tim_st*params.fs:params.both_tim_fn*params.fs].cpu().numpy(),params.fs)
                Csig_in_first_spk_list.append(Csig_in_first_spk)
                Cbak_in_first_spk_list.append(Cbak_in_first_spk)
                Covl_in_first_spk_list.append(Covl_in_first_spk)

                Csig_out_gcn_first_spk,Cbak_out_gcn_first_spk,Covl_out_gcn_first_spk=composite(first_spk_ref[batch_idx,params.both_tim_st*params.fs:params.both_tim_fn*params.fs].cpu().numpy(),y_hat_gcn_first_channel[batch_idx,params.both_tim_st*params.fs:params.both_tim_fn*params.fs].cpu().numpy(),params.fs)
                Csig_out_gcn_first_spk_list.append(Csig_out_gcn_first_spk)
                Cbak_out_gcn_first_spk_list.append(Cbak_out_gcn_first_spk)
                Covl_out_gcn_first_spk_list.append(Covl_out_gcn_first_spk)
            #%% SI-SDR calculation
            if epoc==epoc_for_save and flags.save_oracle_and_noisy:
                si_sdr_out_oracle_first_spk=scale_invariant_signal_distortion_ratio(y_hat_oracle_first_channel[:,params.both_tim_st*params.fs:params.both_tim_fn*params.fs],first_spk_ref[:,params.both_tim_st*params.fs:params.both_tim_fn*params.fs])
                si_sdr_out_noisy_first_spk=scale_invariant_signal_distortion_ratio(y_hat_noisy_first_channel[:,params.both_tim_st*params.fs:params.both_tim_fn*params.fs],first_spk_ref[:,params.both_tim_st*params.fs:params.both_tim_fn*params.fs])
                for batch_idx in range (data.num_graphs):
                    si_sdr_out_oracle_first_spk_list.append(si_sdr_out_oracle_first_spk[batch_idx])
                    si_sdr_out_noisy_first_spk_list.append(si_sdr_out_noisy_first_spk[batch_idx])
            si_sdr_in_first_spk=scale_invariant_signal_distortion_ratio(y_ref[:,params.both_tim_st*params.fs:params.both_tim_fn*params.fs],first_spk_ref[:,params.both_tim_st*params.fs:params.both_tim_fn*params.fs])

            si_sdr_out_gcn_first_spk=scale_invariant_signal_distortion_ratio(y_hat_gcn_first_channel[:,params.both_tim_st*params.fs:params.both_tim_fn*params.fs],first_spk_ref[:,params.both_tim_st*params.fs:params.both_tim_fn*params.fs])
 
            for batch_idx in range (data.num_graphs):
                si_sdr_in_first_spk_list.append(si_sdr_in_first_spk[batch_idx])
                si_sdr_out_gcn_first_spk_list.append(si_sdr_out_gcn_first_spk[batch_idx])
            i=i+1
    #%% save the results to mat files 
    if epoc==epoc_for_save and flags.save_oracle_and_noisy:
        SNR_out_oracle_first_spk,SIR_in_first_spk,SIR_out_oracle_first_spk,STOI_in_first_spk,STOI_out_oracle_first_spk,ESTOI_in_first_spk,ESTOI_out_oracle_first_spk,si_sdr_in_first_spk,si_sdr_out_oracle_first_spk,num_of_examples_per_SNR,NPM_out_oracle_first_spk,SBF_out_oracle_first_spk,Blocking_out_oracle_first_spk,Blocking_with_n_out_oracle_first_spk=check_preformance_compare_measurment(torch.stack(SNR_in_list).cpu().detach().numpy(),torch.stack(SNR_out_oracle_first_spk_list).cpu().detach().numpy(),torch.stack(SNR_in_list).cpu().detach().numpy(),torch.stack(SNR_in_list).cpu().detach().numpy(),STOI_in_first_spk_list,STOI_out_oracle_first_spk_list,ESTOI_in_first_spk_list,ESTOI_out_oracle_first_spk_list,torch.stack(si_sdr_in_first_spk_list).cpu().detach().numpy(),torch.stack(si_sdr_out_oracle_first_spk_list).cpu().detach().numpy(),torch.stack(NPM_out_oracle_first_spk_list).cpu().detach().numpy(),torch.stack(SBF_out_oracle_first_spk_list).cpu().detach().numpy(),torch.stack(Blocking_out_oracle_first_spk_list).cpu().detach().numpy(),torch.stack(Blocking_with_n_out_oracle_first_spk_list).cpu().detach().numpy())
        pesq_in_oracle_first_spk,pesq_out_oracle_first_spk,Csig_in_oracle_first_spk,Csig_out_oracle_first_spk,Cbak_in_oracle_first_spk,Cbak_out_oracle_first_spk,Covl_in_oracle_first_spk,Covl_out_oracle_first_spk,num_of_examples_per_SNR=check_preformance_compare_measurment2(torch.stack(SNR_in_list).cpu().detach().numpy(),pesq_in_first_spk_list,pesq_out_oracle_first_spk_list,Csig_in_first_spk_list,Csig_out_oracle_first_spk_list,Cbak_in_first_spk_list,Cbak_out_oracle_first_spk_list,Covl_in_first_spk_list,Covl_out_oracle_first_spk_list)
        preformace['SNR_out_oracle_first_spk']=np.array(SNR_out_oracle_first_spk)
        preformace['SIR_in_first_spk']=np.array(SIR_in_first_spk)
        preformace['SIR_out_oracle_first_spk']=np.array(SIR_out_oracle_first_spk)
        preformace['STOI_in_first_spk']=np.array(STOI_in_first_spk)
        preformace['STOI_out_oracle_first_spk']=np.array(STOI_out_oracle_first_spk)
        preformace['ESTOI_in_first_spk']=np.array(ESTOI_in_first_spk)
        preformace['ESTOI_out_oracle_first_spk']=np.array(ESTOI_out_oracle_first_spk)
        preformace['si_sdr_in_first_spk']=np.array(si_sdr_in_first_spk)
        preformace['si_sdr_out_oracle_first_spk']=np.array(si_sdr_out_oracle_first_spk)
        preformace['NPM_out_oracle_first_spk']=np.array(NPM_out_oracle_first_spk)
        preformace['SBF_out_oracle_first_spk']=np.array(SBF_out_oracle_first_spk)
        preformace['Blocking_out_oracle_first_spk']=np.array(Blocking_out_oracle_first_spk)
        preformace['Blocking_with_n_out_oracle_first_spk']=np.array(Blocking_with_n_out_oracle_first_spk)
        preformace['pesq_in_oracle_first_spk']=np.array(pesq_in_oracle_first_spk)
        preformace['pesq_out_oracle_first_spk']=np.array(pesq_out_oracle_first_spk)
        preformace['Csig_in_oracle_first_spk']=np.array(Csig_in_oracle_first_spk)
        preformace['Csig_out_oracle_first_spk']=np.array(Csig_out_oracle_first_spk)
        preformace['Cbak_in_oracle_first_spk']=np.array(Cbak_in_oracle_first_spk)
        preformace['Cbak_out_oracle_first_spk']=np.array(Cbak_out_oracle_first_spk)
        preformace['Covl_in_oracle_first_spk']=np.array(Covl_in_oracle_first_spk)
        preformace['Covl_out_oracle_first_spk']=np.array(Covl_out_oracle_first_spk)
        preformace['num_of_examples_per_SNR']=np.array(num_of_examples_per_SNR)
        SNR_out_noisy_first_spk,SIR_in_first_spk,SIR_out_noisy_first_spk,STOI_in_first_spk,STOI_out_noisy_first_spk,ESTOI_in_first_spk,ESTOI_out_noisy_first_spk,si_sdr_in_first_spk,si_sdr_out_noisy_first_spk,num_of_examples_per_SNR,NPM_out_noisy_first_spk,SBF_out_noisy_first_spk,Blocking_out_noisy_first_spk,Blocking_with_n_out_noisy_first_spk=check_preformance_compare_measurment(torch.stack(SNR_in_list).cpu().detach().numpy(),torch.stack(SNR_out_noisy_first_spk_list).cpu().detach().numpy(),torch.stack(SNR_in_list).cpu().detach().numpy(),torch.stack(SNR_in_list).cpu().detach().numpy(),STOI_in_first_spk_list,STOI_out_noisy_first_spk_list,ESTOI_in_first_spk_list,ESTOI_out_noisy_first_spk_list,torch.stack(si_sdr_in_first_spk_list).cpu().detach().numpy(),torch.stack(si_sdr_out_noisy_first_spk_list).cpu().detach().numpy(),torch.stack(NPM_out_noisy_first_spk_list).cpu().detach().numpy(),torch.stack(SBF_out_noisy_first_spk_list).cpu().detach().numpy(),torch.stack(Blocking_out_noisy_first_spk_list).cpu().detach().numpy(),torch.stack(Blocking_with_n_out_noisy_first_spk_list).cpu().detach().numpy())
        pesq_in_noisy_first_spk,pesq_out_noisy_first_spk,Csig_in_noisy_first_spk,Csig_out_noisy_first_spk,Cbak_in_noisy_first_spk,Cbak_out_noisy_first_spk,Covl_in_noisy_first_spk,Covl_out_noisy_first_spk,num_of_examples_per_SNR=check_preformance_compare_measurment2(torch.stack(SNR_in_list).cpu().detach().numpy(),pesq_in_first_spk_list,pesq_out_noisy_first_spk_list,Csig_in_first_spk_list,Csig_out_noisy_first_spk_list,Cbak_in_first_spk_list,Cbak_out_noisy_first_spk_list,Covl_in_first_spk_list,Covl_out_noisy_first_spk_list)
        preformace['SNR_out_noisy_first_spk']=np.array(SNR_out_noisy_first_spk)
        preformace['SIR_out_noisy_first_spk']=np.array(SIR_out_noisy_first_spk)
        preformace['STOI_out_noisy_first_spk']=np.array(STOI_out_noisy_first_spk)
        preformace['ESTOI_out_noisy_first_spk']=np.array(ESTOI_out_noisy_first_spk)
        preformace['si_sdr_out_noisy_first_spk']=np.array(si_sdr_out_noisy_first_spk)
        preformace['NPM_out_noisy_first_spk']=np.array(NPM_out_noisy_first_spk)
        preformace['SBF_out_noisy_first_spk']=np.array(SBF_out_noisy_first_spk)
        preformace['Blocking_out_noisy_first_spk']=np.array(Blocking_out_noisy_first_spk)
        preformace['Blocking_with_n_out_noisy_first_spk']=np.array(Blocking_with_n_out_noisy_first_spk)
        preformace['pesq_in_noisy_first_spk']=np.array(pesq_in_noisy_first_spk)
        preformace['pesq_out_noisy_first_spk']=np.array(pesq_out_noisy_first_spk)
        preformace['Csig_in_noisy_first_spk']=np.array(Csig_in_noisy_first_spk)
        preformace['Csig_out_noisy_first_spk']=np.array(Csig_out_noisy_first_spk)
        preformace['Cbak_in_noisy_first_spk']=np.array(Cbak_in_noisy_first_spk)
        preformace['Cbak_out_noisy_first_spk']=np.array(Cbak_out_noisy_first_spk)
        preformace['Covl_in_noisy_first_spk']=np.array(Covl_in_noisy_first_spk)
        preformace['Covl_out_noisy_first_spk']=np.array(Covl_out_noisy_first_spk)
    SNR_out_gcn_first_spk,SIR_in_first_spk,SIR_out_gcn_first_spk,STOI_in_first_spk,STOI_out_gcn_first_spk,ESTOI_in_first_spk,ESTOI_out_gcn_first_spk,si_sdr_in_first_spk,si_sdr_out_gcn_first_spk,num_of_examples_per_SNR,NPM_out_gcn_first_spk,SBF_out_gcn_first_spk,Blocking_out_gcn_first_spk,Blocking_with_n_out_gcn_first_spk=check_preformance_compare_measurment(torch.stack(SNR_in_list).cpu().detach().numpy(),torch.stack(SNR_out_gcn_first_spk_list).cpu().detach().numpy(),torch.stack(SNR_in_list).cpu().detach().numpy(),torch.stack(SNR_in_list).cpu().detach().numpy(),STOI_in_first_spk_list,STOI_out_gcn_first_spk_list,ESTOI_in_first_spk_list,ESTOI_out_gcn_first_spk_list,torch.stack(si_sdr_in_first_spk_list).cpu().detach().numpy(),torch.stack(si_sdr_out_gcn_first_spk_list).cpu().detach().numpy(),torch.stack(NPM_out_gcn_first_spk_list).cpu().detach().numpy(),torch.stack(SBF_out_gcn_first_spk_list).cpu().detach().numpy(),torch.stack(Blocking_out_gcn_first_spk_list).cpu().detach().numpy(),torch.stack(Blocking_with_n_out_gcn_first_spk_list).cpu().detach().numpy())
    pesq_in_gcn_first_spk,pesq_out_gcn_first_spk,Csig_in_gcn_first_spk,Csig_out_gcn_first_spk,Cbak_in_gcn_first_spk,Cbak_out_gcn_first_spk,Covl_in_gcn_first_spk,Covl_out_gcn_first_spk,num_of_examples_per_SNR=check_preformance_compare_measurment2(torch.stack(SNR_in_list).cpu().detach().numpy(),pesq_in_first_spk_list,pesq_out_gcn_first_spk_list,Csig_in_first_spk_list,Csig_out_gcn_first_spk_list,Cbak_in_first_spk_list,Cbak_out_gcn_first_spk_list,Covl_in_first_spk_list,Covl_out_gcn_first_spk_list)
    preformace['SNR_out_gcn_first_spk']=np.array(SNR_out_gcn_first_spk)
    preformace['SIR_out_gcn_first_spk']=np.array(SIR_out_gcn_first_spk)
    preformace['STOI_out_gcn_first_spk']=np.array(STOI_out_gcn_first_spk)
    preformace['ESTOI_out_gcn_first_spk']=np.array(ESTOI_out_gcn_first_spk)
    preformace['si_sdr_out_gcn_first_spk']=np.array(si_sdr_out_gcn_first_spk)
    preformace['NPM_out_gcn_first_spk']=np.array(NPM_out_gcn_first_spk)
    preformace['SBF_out_gcn_first_spk']=np.array(SBF_out_gcn_first_spk)
    preformace['Blocking_out_gcn_first_spk']=np.array(Blocking_out_gcn_first_spk)
    preformace['Blocking_with_n_out_gcn_first_spk']=np.array(Blocking_with_n_out_gcn_first_spk)
    preformace['pesq_in_gcn_first_spk']=np.array(pesq_in_gcn_first_spk)
    preformace['pesq_out_gcn_first_spk']=np.array(pesq_out_gcn_first_spk)
    preformace['Csig_in_gcn_first_spk']=np.array(Csig_in_gcn_first_spk)
    preformace['Csig_out_gcn_first_spk']=np.array(Csig_out_gcn_first_spk)
    preformace['Cbak_in_gcn_first_spk']=np.array(Cbak_in_gcn_first_spk)
    preformace['Cbak_out_gcn_first_spk']=np.array(Cbak_out_gcn_first_spk)
    preformace['Covl_in_gcn_first_spk']=np.array(Covl_in_gcn_first_spk)
    preformace['Covl_out_gcn_first_spk']=np.array(Covl_out_gcn_first_spk)
    if epoc==epoc_for_save and flags.save_oracle_and_noisy:
        sio.savemat(results_path+'measurments_on_ref_oracle_epoc_'+str(epoc)+'.mat', preformace)
    else:
        sio.savemat(results_path+'measurments_on_ref_epoc_'+str(epoc)+'.mat', preformace)