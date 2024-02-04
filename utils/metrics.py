
import numpy as np
import torch
from torch import Tensor

def NPM(estimated_RTFs, oracle_RTFs):
    npm = 0
    mic = [0, 1, 3, 4]
    
    # Remove singleton dimensions for simplicity
    oracle_RTFs = oracle_RTFs.squeeze()
    estimated_RTFs = estimated_RTFs.squeeze()
    
    # Loop over the specified microphones
    for m in mic:
        # Check if RTFs has 2 dimensions
        if len(oracle_RTFs.shape) == 2:
            # Calculate epsilon for a single microphone
            epsilon = oracle_RTFs[:, m] - ((oracle_RTFs[:, m].conj() @ estimated_RTFs[:, m]) /
                                           ((estimated_RTFs[:, m].conj() @ estimated_RTFs[:, m]) + 1e-20) * estimated_RTFs[:, m])
            
            # Accumulate the NPM loss for the current microphone
            npm += 10 * torch.log10(torch.norm(epsilon) / (torch.norm(oracle_RTFs[:, m]) + 1e-20))
        else:
            # Calculate epsilon for batch of samples
            epsilon = oracle_RTFs[:, :, m] - ((torch.diagonal((oracle_RTFs[:, :, m].conj() @ estimated_RTFs[:, :, m].T)) /
                                                (torch.diagonal(estimated_RTFs[:, :, m].conj() @ estimated_RTFs[:, :, m].T) + 1e-20)) *
                                               estimated_RTFs[:, :, m].conj().T).T
            
            # Accumulate the NPM loss for the current microphone  
            npm += 10 * torch.log10(torch.norm(epsilon, dim=1) / (torch.norm(oracle_RTFs[:, :, m], dim=1)) + 1e-20)

    # Calculate the mean NPM loss across specified microphones
    return npm / len(mic)

def SBF(estimated_RTFs, oracle_RTFs, ref):
    sbf = 0
    mic = [0, 1, 3, 4]
    # Loop over the specified microphones
    for m in mic:
        # Check if the reference signal has only one dimension(no batch dimension)
        if len(ref.shape) == 1:
            oracle_RTF = oracle_RTFs[:, m]
            estimated_RTF = estimated_RTFs[:, m]
            
            # Calculate the difference for the signal part (r_n)
            r_n = torch.nn.functional.conv1d(ref.unsqueeze(0), torch.flip(oracle_RTF, (0,)).view(1, 1, -1)).squeeze() - torch.nn.functional.conv1d(ref.unsqueeze(0), torch.flip(estimated_RTF, (0,)).view(1, 1, -1)).squeeze()
            
            # Calculate the difference for the noise part (ref_filter)
            ref_filter = torch.nn.functional.conv1d(ref.unsqueeze(0), torch.flip(oracle_RTF, (0,)).view(1, 1, -1)).squeeze()
            
            # Accumulate the SBF loss for the current microphone
            sbf += 10 * torch.log10(torch.var(ref_filter) / (torch.var(r_n) + 1e-20) + 1e-20)
        else:
            # Calculate the difference for the signal part (r_n) and noise part (ref_filter) for batch of samples
            r_n = torch.nn.functional.conv1d(ref.unsqueeze(1), torch.flip(oracle_RTFs[:, m], (0,)).view(1, 1, -1)).squeeze() - torch.nn.functional.conv1d(ref.unsqueeze(1), torch.flip(estimated_RTFs[:, m], (0,)).view(1, 1, -1)).squeeze()
            ref_filter = torch.nn.functional.conv1d(ref.unsqueeze(1), torch.flip(oracle_RTFs[:, m], (0,)).view(1, 1, -1)).squeeze()
            
            # Accumulate the SBF loss for the current set of microphones
            sbf += 10 * torch.log10(torch.var(ref_filter, dim=1) / (torch.var(r_n, dim=1) + 1e-20) + 1e-20)
    
    # Calculate the mean SBF loss across specified microphones
    return sbf / len(mic)

def Blocking_with_n(estimated_RTFs, ref_s, mic_s, ref_n, mic_n):
    Blocking_with_n = 0
    
    # Create a dirac tensor with ones at the center position of the estimated RTFs
    dirac = torch.zeros(estimated_RTFs.shape[0], estimated_RTFs.shape[1]).to(estimated_RTFs.device)
    dirac[:, estimated_RTFs.shape[1] // 2] = 1
    
    mic = [0, 1, 3, 4]
    
    # Loop over the specified microphones
    for m in mic:
        # Check if the current microphone index is greater than 2(reference microphone)
        if m > 2:
            # Calculate the difference for the signal part (s) and noise part (n) for the current microphone
            s = torch.nn.functional.conv1d(mic_s[:, :, m - 1].unsqueeze(1), torch.flip(dirac, (0,)).view(1, 1, -1)).squeeze() - torch.nn.functional.conv1d(ref_s.unsqueeze(1), torch.flip(torch.fft.ifftshift(estimated_RTFs[:, :, m], -1), (0,)).view(1, 1, -1)).squeeze()
            n = torch.nn.functional.conv1d(mic_n[:, :, m - 1].unsqueeze(1), torch.flip(dirac, (0,)).view(1, 1, -1)).squeeze() - torch.nn.functional.conv1d(ref_n.unsqueeze(1), torch.flip(torch.fft.ifftshift(estimated_RTFs[:, :, m], -1), (0,)).view(1, 1, -1)).squeeze()
        else:
            # Calculate the difference for the signal part (s) and noise part (n) for the current microphone
            s = torch.nn.functional.conv1d(mic_s[:, :, m].unsqueeze(1), torch.flip(dirac, (0,)).view(1, 1, -1)).squeeze() - torch.nn.functional.conv1d(ref_s.unsqueeze(1), torch.flip(torch.fft.ifftshift(estimated_RTFs[:, :, m], -1), (0,)).view(1, 1, -1)).squeeze()
            n = torch.nn.functional.conv1d(mic_n[:, :, m].unsqueeze(1), torch.flip(dirac, (0,)).view(1, 1, -1)).squeeze() - torch.nn.functional.conv1d(ref_n.unsqueeze(1), torch.flip(torch.fft.ifftshift(estimated_RTFs[:, :, m], -1), (0,)).view(1, 1, -1)).squeeze()
        
        # Accumulate the Blocking_with_n loss for the current microphone
        Blocking_with_n += 10 * torch.log10(torch.var(n, dim=1) / (torch.var(s, dim=1) + 1e-20) + 1e-20)
    
    # Calculate the mean Blocking_with_n loss across specified microphones
    return Blocking_with_n / len(mic)

def Blocking(estimated_RTFs, ref_s, mic_s):
    mic = [0, 1, 3, 4]
    blocking = 0
    
    # Use the entire signal for 's'
    s = mic_s
    
    # Create a dirac tensor with ones at the center position of the estimated RTFs
    dirac = torch.zeros(estimated_RTFs.shape[0], estimated_RTFs.shape[1]).to(estimated_RTFs.device)
    dirac[:, estimated_RTFs.shape[1] // 2] = 1
    
    # Loop over the specified microphones
    for m in mic:
        # Check if the current microphone index is greater than 2(reference microphone)
        if m > 2:
            # Calculate the difference for the the signal at the m'th microphone (f) and the signal part (r_n) for the reference microphone convolved with the estimated RTF
            f = torch.nn.functional.conv1d(mic_s[:, :, m - 1].unsqueeze(1), torch.flip(dirac, (0,)).view(1, 1, -1)).squeeze()
            se = torch.nn.functional.conv1d(ref_s.unsqueeze(1), torch.flip(torch.fft.ifftshift(estimated_RTFs[:, :, m], -1), (0,)).view(1, 1, -1)).squeeze()
            r_n = f - se
            
            # Accumulate the Blocking loss for the current microphone
            blocking += 10 * torch.log10(torch.var(s[:, :, m - 1], dim=1) / (torch.var(r_n, dim=1) + 1e-20) + 1e-20)
        else:
            # Calculate the difference for the the signal at the m'th microphone (f) and the signal part (r_n) for the reference microphone convolved with the estimated RTF
            f = torch.nn.functional.conv1d(mic_s[:, :, m].unsqueeze(1), torch.flip(dirac, (0,)).view(1, 1, -1)).squeeze()
            se = torch.nn.functional.conv1d(ref_s.unsqueeze(1), torch.flip(torch.fft.ifftshift(estimated_RTFs[:, :, m], -1), (0,)).view(1, 1, -1)).squeeze()
            r_n = f - se
            
            # Accumulate the Blocking loss for the current microphone
            blocking += 10 * torch.log10(torch.var(s[:, :, m], dim=1) / (torch.var(r_n, dim=1) + 1e-20) + 1e-20)
    
    # Calculate the mean Blocking loss across specified microphones
    return blocking / len(mic)

def _check_same_shape(preds: Tensor, target: Tensor) -> None:
    """Check that predictions and target have the same shape, else raise error."""
    if preds.shape != target.shape:
        raise RuntimeError(f"Predictions and targets are expected to have the same shape, pred has shape of {preds.shape} and target has shape of {target.shape}")

def scale_invariant_signal_distortion_ratio(preds: Tensor, target: Tensor, zero_mean: bool = True) -> Tensor:
    """Calculates Scale-invariant signal-to-distortion ratio (SI-SDR) metric. The SI-SDR value is in general
    considered an overall measure of how good a source sound.
    Args:
        preds:
            shape ``[...,time]``
        target:
            shape ``[...,time]``
        zero_mean:
            If to zero mean target and preds or not
    Returns:
        si-sdr value of shape [...]
    Example:
        #>>> from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio
        #>>> target = torch.tensor([3.0, -0.5, 2.0, 7.0])
        #>>> preds = torch.tensor([2.5, 0.0, 2.0, 8.0])
        #>>> scale_invariant_signal_distortion_ratio(preds, target)
        tensor(18.4030)
    References:
        [1] Le Roux, Jonathan, et al. "SDR half-baked or well done." IEEE International Conference on Acoustics, Speech
        and Signal Processing (ICASSP) 2019.
    """
    #print(f"shape preds: {preds.shape} \nshape target: {target.shape}")
    _check_same_shape(preds, target)
    EPS = torch.finfo(preds.dtype).eps

    if zero_mean:
        target = target - torch.mean(target, dim=-1, keepdim=True)
        preds = preds - torch.mean(preds, dim=-1, keepdim=True)

    alpha = (torch.sum(preds * target, dim=-1, keepdim=True) + EPS) / (
        torch.sum(target ** 2, dim=-1, keepdim=True) + EPS
    )
    target_scaled = alpha * target

    noise = target_scaled - preds

    val = (torch.sum(target_scaled ** 2, dim=-1) + EPS) / (torch.sum(noise ** 2, dim=-1) + EPS)
    val = 10 * torch.log10(val)
    
    return val

def create_Qvv_k_batch(Y_STFT_matrix):
    '''
    calculate Qvv for each batch and each frequency point (k) in the STFT domain 
    Y_STFT_matrix: (batch_size,frame_count,M)
    '''  
    Qvv=torch.bmm(Y_STFT_matrix.permute(0, 2, 1),Y_STFT_matrix.conj())/len(Y_STFT_matrix[0,:,0])
    return Qvv

def MVDR_RTFs_batchs(first_spk, y, noise,H,args,device,batch_size):
    '''
    calculate the MVDR weights and return the output of the beamformer for the clean speech, the noisy speech and the noise
    first_spk: (batch_size,frame_count,M)
    y: (batch_size,frame_count,M)
    noise: (batch_size,frame_count,M)
    H: (batch_size,NUP,M)
    args: args
    device: device
    batch_size: batch_size
    
    '''
    win = torch.hamming_window(args.wlen).to(device)
    e=1e-6
    eye_M=torch.eye(args.M).repeat(batch_size, 1, 1).to(device)
    frame_count = 1 + (y.shape[1] - args.wlen ) //args.n_hop
    Y_STFT_matrix=torch.zeros((batch_size,int(eval(args.NUP)),frame_count,args.M),dtype=torch.cfloat).to(device)
    first_spk_STFT_matrix=torch.zeros((batch_size,int(eval(args.NUP)),frame_count,args.M),dtype=torch.cfloat).to(device)
    N_STFT_matrix=torch.zeros((batch_size,int(eval(args.NUP)),frame_count,args.M),dtype=torch.cfloat).to(device)

    for m in range(args.M):
        Y_STFT_matrix[:,:,:,m]=torch.stft(y[:,:,m],args.NFFT,args.n_hop,args.wlen,win,center=False,return_complex=True)
        first_spk_STFT_matrix[:,:,:,m]=torch.stft(first_spk[:,:,m],args.NFFT,args.n_hop,args.wlen,win,center=False,return_complex=True)
        N_STFT_matrix[:,:,:,m]=torch.stft(noise[:,:,m],args.NFFT,args.n_hop,args.wlen,win,center=False,return_complex=True)
    output_y_stft = torch.zeros(batch_size,int(eval(args.NUP)),frame_count, dtype=torch.cfloat).to(device)
    output_first_spk_stft = torch.zeros(batch_size,int(eval(args.NUP)),frame_count, dtype=torch.cfloat).to(device)
    output_n_stft = torch.zeros(batch_size,int(eval(args.NUP)),frame_count, dtype=torch.cfloat).to(device)

    for f in range(int(eval(args.NUP))):
        
        Qvv=create_Qvv_k_batch(Y_STFT_matrix[:,f,:frame_count//5,:])
        H_k = torch.unsqueeze(H[:,f,:],2)
        # calculate the weights for each frequency point (k) in the STFT domain
        inv_qvv = torch.inverse(Qvv+e*torch.norm(Qvv,dim=(1,2))[:, None, None]*eye_M)
        b = torch.bmm(inv_qvv,H_k)     
        inv_temp = torch.squeeze(torch.bmm(H_k.conj().permute(0, 2, 1) , b)) + e*torch.norm(torch.bmm(H_k.conj().permute(0, 2, 1) , b),dim=(1,2))
        w =(torch.squeeze(b).T/inv_temp).T
        # calculate the output of the beamformer for each frequency point (k) in the STFT domain
        output_y_stft[:,f,:]=torch.squeeze(torch.bmm(torch.unsqueeze(w.conj(),1) , torch.squeeze(Y_STFT_matrix[:,f,:,:]).permute(0, 2, 1)))
        output_first_spk_stft[:,f,:]=torch.squeeze(torch.bmm(torch.unsqueeze(w.conj(),1) , torch.squeeze(first_spk_STFT_matrix[:,f,:,:]).permute(0, 2, 1)))
        output_n_stft[:,f,:]=torch.squeeze(torch.bmm(torch.unsqueeze(w.conj(),1) , torch.squeeze(N_STFT_matrix[:,f,:,:]).permute(0, 2, 1)))
    return output_y_stft, output_first_spk_stft, output_n_stft

def si_sdr_torchaudio_calc(estimate, reference, epsilon=1e-8):
        '''
        calculate si_sdr - scale-invariant signal-to-distortion ratio 
        estimate: (batch_size,frame_count)
        reference: (batch_size,frame_count)
        '''
        estimate = estimate - estimate.mean()
        reference = reference - reference.mean()
        reference_pow = reference.pow(2).mean()
        mix_pow = (estimate * reference).mean()
        scale = mix_pow / (reference_pow + epsilon)
        reference = scale * reference
        error = estimate - reference
        reference_pow = reference.pow(2)
        error_pow = error.pow(2)
        reference_pow = reference_pow.mean()
        error_pow = error_pow.mean()
        sisdr = 10 * torch.log10(reference_pow) - 10 * torch.log10(error_pow)
        return sisdr

def check_preformance_compare_measurment(SNR_in,SNR_out,SIR_in,SIR_out,STOI_in,STOI_out,ESTOI_in,ESTOI_out,si_sdr_in,si_sdr_out,NPM_out,SBF_out,Blocking_out,Blocking_with_n_out):
    '''
    check the preformance of the model for each SNR
    SNR_in: SNR of the input
    SNR_out: SNR of the output
    SIR_in: SIR of the input
    SIR_out: SIR of the output
    STOI_in: STOI of the input
    STOI_out: STOI of the output
    ESTOI_in: ESTOI of the input
    ESTOI_out: ESTOI of the output
    si_sdr_in: si_sdr of the input
    si_sdr_out: si_sdr of the output
    NPM_out: NPM of the output
    SBF_out: SBF of the output
    Blocking_out: Blocking of the output
    Blocking_with_n_out: Blocking_with_n of the output
    '''
    SNR_m_10_out,SNR_m_6_out,SNR_m_2_out=[],[],[]
    SNR_p_2_out,SNR_p_6_out,SNR_p_10_out=[],[],[]
    SIR_m_10_in,SIR_m_6_in,SIR_m_2_in=[],[],[]
    SIR_p_2_in,SIR_p_6_in,SIR_p_10_in=[],[],[]    
    SIR_m_10_out,SIR_m_6_out,SIR_m_2_out=[],[],[]
    SIR_p_2_out,SIR_p_6_out,SIR_p_10_out=[],[],[]
    STOI_m_10_in,STOI_m_6_in,STOI_m_2_in=[],[],[]
    STOI_p_2_in,STOI_p_6_in,STOI_p_10_in=[],[],[]    
    STOI_m_10_out,STOI_m_6_out,STOI_m_2_out=[],[],[]
    STOI_p_2_out,STOI_p_6_out,STOI_p_10_out=[],[],[]
    ESTOI_m_10_in,ESTOI_m_6_in,ESTOI_m_2_in=[],[],[]
    ESTOI_p_2_in,ESTOI_p_6_in,ESTOI_p_10_in=[],[],[]
    ESTOI_m_10_out,ESTOI_m_6_out,ESTOI_m_2_out=[],[],[]
    ESTOI_p_2_out,ESTOI_p_6_out,ESTOI_p_10_out=[],[],[]
    si_sdr_m_10_in,si_sdr_m_6_in,si_sdr_m_2_in=[],[],[]
    si_sdr_p_2_in,si_sdr_p_6_in,si_sdr_p_10_in=[],[],[]
    si_sdr_m_10_out,si_sdr_m_6_out,si_sdr_m_2_out=[],[],[]
    si_sdr_p_2_out,si_sdr_p_6_out,si_sdr_p_10_out=[],[],[]  
    NPM_m_10_in,NPM_m_6_in,NPM_m_2_in=[],[],[]
    NPM_p_2_in,NPM_p_6_in,NPM_p_10_in=[],[],[]
    SBF_m_10_in,SBF_m_6_in,SBF_m_2_in=[],[],[]
    SBF_p_2_in,SBF_p_6_in,SBF_p_10_in=[],[],[]
    Blocking_m_10_in,Blocking_m_6_in,Blocking_m_2_in=[],[],[]
    Blocking_p_2_in,Blocking_p_6_in,Blocking_p_10_in=[],[],[]
    Blocking_with_n_m_10_out,Blocking_with_n_m_6_out,Blocking_with_n_m_2_out=[],[],[]
    Blocking_with_n_p_2_out,Blocking_with_n_p_6_out,Blocking_with_n_p_10_out=[],[],[]
    num_of_examples_per_SNR=np.zeros(6)
    res = [int(np.round(x)) for x in SNR_in]
    for i in range(len(res)):
        if res[i]==-10:
            SNR_m_10_out.append(SNR_out[i])
            SIR_m_10_in.append(SIR_in[i])
            SIR_m_10_out.append(SIR_out[i])
            STOI_m_10_in.append(STOI_in[i])
            STOI_m_10_out.append(STOI_out[i])
            ESTOI_m_10_in.append(ESTOI_in[i])
            ESTOI_m_10_out.append(ESTOI_out[i])
            si_sdr_m_10_in.append(si_sdr_in[i])
            si_sdr_m_10_out.append(si_sdr_out[i])
            NPM_m_10_in.append(NPM_out[i])
            SBF_m_10_in.append(SBF_out[i])
            Blocking_m_10_in.append(Blocking_out[i])
            Blocking_with_n_m_10_out.append(Blocking_with_n_out[i])
            num_of_examples_per_SNR[0]+=1

        elif res[i]==-6:
            SNR_m_6_out.append(SNR_out[i])
            SIR_m_6_in.append(SIR_in[i])
            SIR_m_6_out.append(SIR_out[i])
            STOI_m_6_in.append(STOI_in[i])
            STOI_m_6_out.append(STOI_out[i])
            ESTOI_m_6_in.append(ESTOI_in[i])
            ESTOI_m_6_out.append(ESTOI_out[i])
            si_sdr_m_6_in.append(si_sdr_in[i])
            si_sdr_m_6_out.append(si_sdr_out[i])
            NPM_m_6_in.append(NPM_out[i])
            SBF_m_6_in.append(SBF_out[i])
            Blocking_m_6_in.append(Blocking_out[i])
            Blocking_with_n_m_6_out.append(Blocking_with_n_out[i])
            num_of_examples_per_SNR[1]+=1

        elif res[i]==-2:
            SNR_m_2_out.append(SNR_out[i])
            SIR_m_2_in.append(SIR_in[i])
            SIR_m_2_out.append(SIR_out[i])
            STOI_m_2_in.append(STOI_in[i])
            STOI_m_2_out.append(STOI_out[i])
            ESTOI_m_2_in.append(ESTOI_in[i])
            ESTOI_m_2_out.append(ESTOI_out[i])
            si_sdr_m_2_in.append(si_sdr_in[i])
            si_sdr_m_2_out.append(si_sdr_out[i])       
            NPM_m_2_in.append(NPM_out[i])      
            SBF_m_2_in.append(SBF_out[i])
            Blocking_m_2_in.append(Blocking_out[i])
            Blocking_with_n_m_2_out.append(Blocking_with_n_out[i])       
            num_of_examples_per_SNR[2]+=1
        elif res[i]==2:
            SNR_p_2_out.append(SNR_out[i])
            SIR_p_2_in.append(SIR_in[i])
            SIR_p_2_out.append(SIR_out[i])
            STOI_p_2_in.append(STOI_in[i])
            STOI_p_2_out.append(STOI_out[i])
            ESTOI_p_2_in.append(ESTOI_in[i])
            ESTOI_p_2_out.append(ESTOI_out[i])
            si_sdr_p_2_in.append(si_sdr_in[i])
            si_sdr_p_2_out.append(si_sdr_out[i])
            NPM_p_2_in.append(NPM_out[i])
            SBF_p_2_in.append(SBF_out[i])
            Blocking_p_2_in.append(Blocking_out[i])
            Blocking_with_n_p_2_out.append(Blocking_with_n_out[i])
            
            num_of_examples_per_SNR[3]+=1 
        elif res[i]==6:
            SNR_p_6_out.append(SNR_out[i])
            SIR_p_6_in.append(SIR_in[i])
            SIR_p_6_out.append(SIR_out[i])
            STOI_p_6_in.append(STOI_in[i])
            STOI_p_6_out.append(STOI_out[i])
            ESTOI_p_6_in.append(ESTOI_in[i])
            ESTOI_p_6_out.append(ESTOI_out[i])
            si_sdr_p_6_in.append(si_sdr_in[i])
            si_sdr_p_6_out.append(si_sdr_out[i])
            NPM_p_6_in.append(NPM_out[i])
            SBF_p_6_in.append(SBF_out[i])
            Blocking_p_6_in.append(Blocking_out[i])
            Blocking_with_n_p_6_out.append(Blocking_with_n_out[i])
            num_of_examples_per_SNR[4]+=1
        elif res[i]==10:
            SNR_p_10_out.append(SNR_out[i])
            SIR_p_10_in.append(SIR_in[i])
            SIR_p_10_out.append(SIR_out[i])
            STOI_p_10_in.append(STOI_in[i])
            STOI_p_10_out.append(STOI_out[i])
            ESTOI_p_10_in.append(ESTOI_in[i])
            ESTOI_p_10_out.append(ESTOI_out[i])
            si_sdr_p_10_in.append(si_sdr_in[i])
            si_sdr_p_10_out.append(si_sdr_out[i])
            NPM_p_10_in.append(NPM_out[i])
            SBF_p_10_in.append(SBF_out[i])
            Blocking_p_10_in.append(Blocking_out[i])
            Blocking_with_n_p_10_out.append(Blocking_with_n_out[i])
            num_of_examples_per_SNR[5]+=1
        else:
            print("out of range")
    SNR_out=[np.mean(SNR_m_10_out),np.mean(SNR_m_6_out),np.mean(SNR_m_2_out),np.mean(SNR_p_2_out),np.mean(SNR_p_6_out),np.mean(SNR_p_10_out)]
    SIR_in=[np.mean(SIR_m_10_in),np.mean(SIR_m_6_in),np.mean(SIR_m_2_in),np.mean(SIR_p_2_in),np.mean(SIR_p_6_in),np.mean(SIR_p_10_in)]
    SIR_out=[np.mean(SIR_m_10_out),np.mean(SIR_m_6_out),np.mean(SIR_m_2_out),np.mean(SIR_p_2_out),np.mean(SIR_p_6_out),np.mean(SIR_p_10_out)]
    STOI_in=[np.mean(STOI_m_10_in),np.mean(STOI_m_6_in),np.mean(STOI_m_2_in),np.mean(STOI_p_2_in),np.mean(STOI_p_6_in),np.mean(STOI_p_10_in)]
    STOI_out=[np.mean(STOI_m_10_out),np.mean(STOI_m_6_out),np.mean(STOI_m_2_out),np.mean(STOI_p_2_out),np.mean(STOI_p_6_out),np.mean(STOI_p_10_out)]
    ESTOI_in=[np.mean(ESTOI_m_10_in),np.mean(ESTOI_m_6_in),np.mean(ESTOI_m_2_in),np.mean(ESTOI_p_2_in),np.mean(ESTOI_p_6_in),np.mean(ESTOI_p_10_in)]
    ESTOI_out=[np.mean(ESTOI_m_10_out),np.mean(ESTOI_m_6_out),np.mean(ESTOI_m_2_out),np.mean(ESTOI_p_2_out),np.mean(ESTOI_p_6_out),np.mean(ESTOI_p_10_out)]
    si_sdr_in=[np.mean(si_sdr_m_10_in),np.mean(si_sdr_m_6_in),np.mean(si_sdr_m_2_in),np.mean(si_sdr_p_2_in),np.mean(si_sdr_p_6_in),np.mean(si_sdr_p_10_in)]
    si_sdr_out=[np.mean(si_sdr_m_10_out),np.mean(si_sdr_m_6_out),np.mean(si_sdr_m_2_out),np.mean(si_sdr_p_2_out),np.mean(si_sdr_p_6_out),np.mean(si_sdr_p_10_out)]
    NPM_out=[np.mean(NPM_m_10_in),np.mean(NPM_m_6_in),np.mean(NPM_m_2_in),np.mean(NPM_p_2_in),np.mean(NPM_p_6_in),np.mean(NPM_p_10_in)]
    SBF_out=[np.mean(SBF_m_10_in),np.mean(SBF_m_6_in),np.mean(SBF_m_2_in),np.mean(SBF_p_2_in),np.mean(SBF_p_6_in),np.mean(SBF_p_10_in)]
    Blocking_out=[np.mean(Blocking_m_10_in),np.mean(Blocking_m_6_in),np.mean(Blocking_m_2_in),np.mean(Blocking_p_2_in),np.mean(Blocking_p_6_in),np.mean(Blocking_p_10_in)]
    Blocking_with_n_out=[np.mean(Blocking_with_n_m_10_out),np.mean(Blocking_with_n_m_6_out),np.mean(Blocking_with_n_m_2_out),np.mean(Blocking_with_n_p_2_out),np.mean(Blocking_with_n_p_6_out),np.mean(Blocking_with_n_p_10_out)]
    return SNR_out,SIR_in,SIR_out,STOI_in,STOI_out,ESTOI_in,ESTOI_out,si_sdr_in,si_sdr_out,num_of_examples_per_SNR,NPM_out,SBF_out,Blocking_out,Blocking_with_n_out

def check_preformance_compare_measurment2(SNR_in,pesq_in,pesq_out,Csig_in,Csig_out,Cbak_in,Cbak_out,Covl_in,Covl_out):
    '''
    check the preformance of the model for each SNR
    SNR_in: SNR of the input
    pesq_in: pesq of the input
    pesq_out: pesq of the output
    Csig_in: Csig of the input
    Csig_out: Csig of the output
    Cbak_in: Cbak of the input
    Cbak_out: Cbak of the output
    Covl_in: Covl of the input
    Covl_out: Covl of the output
    '''
    pesq_m_10_in,pesq_m_6_in,pesq_m_2_in=[],[],[]
    pesq_p_2_in,pesq_p_6_in,pesq_p_10_in=[],[],[]    
    pesq_m_10_out,pesq_m_6_out,pesq_m_2_out=[],[],[]
    pesq_p_2_out,pesq_p_6_out,pesq_p_10_out=[],[],[]
    Csig_m_10_in,Csig_m_6_in,Csig_m_2_in=[],[],[]
    Csig_p_2_in,Csig_p_6_in,Csig_p_10_in=[],[],[]
    Csig_m_10_out,Csig_m_6_out,Csig_m_2_out=[],[],[]
    Csig_p_2_out,Csig_p_6_out,Csig_p_10_out=[],[],[]  
    Cbak_m_10_in,Cbak_m_6_in,Cbak_m_2_in=[],[],[]
    Cbak_p_2_in,Cbak_p_6_in,Cbak_p_10_in=[],[],[] 
    Cbak_m_10_out,Cbak_m_6_out,Cbak_m_2_out=[],[],[]
    Cbak_p_2_out,Cbak_p_6_out,Cbak_p_10_out=[],[],[]  
    Covl_m_10_in,Covl_m_6_in,Covl_m_2_in=[],[],[]
    Covl_p_2_in,Covl_p_6_in,Covl_p_10_in=[],[],[]
    Covl_m_10_out,Covl_m_6_out,Covl_m_2_out=[],[],[]
    Covl_p_2_out,Covl_p_6_out,Covl_p_10_out=[],[],[]  
        
    num_of_examples_per_SNR=np.zeros(6)
    res = [int(np.round(x)) for x in SNR_in]
    for i in range(len(res)):
        if res[i]==-10:
            pesq_m_10_in.append(pesq_in[i])
            pesq_m_10_out.append(pesq_out[i])
            Csig_m_10_in.append(Csig_in[i])
            Csig_m_10_out.append(Csig_out[i])
            Cbak_m_10_in.append(Cbak_in[i])
            Cbak_m_10_out.append(Cbak_out[i])
            Covl_m_10_in.append(Covl_in[i])
            Covl_m_10_out.append(Covl_out[i])

            num_of_examples_per_SNR[0]+=1

        elif res[i]==-6:
            pesq_m_6_in.append(pesq_in[i])
            pesq_m_6_out.append(pesq_out[i])
            Csig_m_6_in.append(Csig_in[i])
            Csig_m_6_out.append(Csig_out[i])
            Cbak_m_6_in.append(Cbak_in[i])
            Cbak_m_6_out.append(Cbak_out[i])
            Covl_m_6_in.append(Covl_in[i])
            Covl_m_6_out.append(Covl_out[i])
            
            num_of_examples_per_SNR[1]+=1

        elif res[i]==-2:
            pesq_m_2_in.append(pesq_in[i])
            pesq_m_2_out.append(pesq_out[i])

            Csig_m_2_in.append(Csig_in[i])
            Csig_m_2_out.append(Csig_out[i])       
            Cbak_m_2_in.append(Cbak_in[i])
            Cbak_m_2_out.append(Cbak_out[i])
            Covl_m_2_in.append(Covl_in[i])
            Covl_m_2_out.append(Covl_out[i])
            
            num_of_examples_per_SNR[2]+=1
        elif res[i]==2:
            pesq_p_2_in.append(pesq_in[i])
            pesq_p_2_out.append(pesq_out[i])

            Csig_p_2_in.append(Csig_in[i])
            Csig_p_2_out.append(Csig_out[i])
            Cbak_p_2_in.append(Cbak_in[i])
            Cbak_p_2_out.append(Cbak_out[i])
            Covl_p_2_in.append(Covl_in[i])
            Covl_p_2_out.append(Covl_out[i])
            
            num_of_examples_per_SNR[3]+=1 
        elif res[i]==6:
            pesq_p_6_in.append(pesq_in[i])
            pesq_p_6_out.append(pesq_out[i])

            Csig_p_6_in.append(Csig_in[i])
            Csig_p_6_out.append(Csig_out[i])
            Cbak_p_6_in.append(Cbak_in[i])
            Cbak_p_6_out.append(Cbak_out[i])
            Covl_p_6_in.append(Covl_in[i])
            Covl_p_6_out.append(Covl_out[i])
           
            num_of_examples_per_SNR[4]+=1
        elif res[i]==10:
            pesq_p_10_in.append(pesq_in[i])
            pesq_p_10_out.append(pesq_out[i])

            Csig_p_10_in.append(Csig_in[i])
            Csig_p_10_out.append(Csig_out[i])
            Cbak_p_10_in.append(Cbak_in[i])
            Cbak_p_10_out.append(Cbak_out[i])
            Covl_p_10_in.append(Covl_in[i])
            Covl_p_10_out.append(Covl_out[i])
           
            num_of_examples_per_SNR[5]+=1
        else:
            print("out of range")
    pesq_in=[np.mean(pesq_m_10_in),np.mean(pesq_m_6_in),np.mean(pesq_m_2_in),np.mean(pesq_p_2_in),np.mean(pesq_p_6_in),np.mean(pesq_p_10_in)]
    pesq_out=[np.mean(pesq_m_10_out),np.mean(pesq_m_6_out),np.mean(pesq_m_2_out),np.mean(pesq_p_2_out),np.mean(pesq_p_6_out),np.mean(pesq_p_10_out)]
    Csig_in=[np.mean(Csig_m_10_in),np.mean(Csig_m_6_in),np.mean(Csig_m_2_in),np.mean(Csig_p_2_in),np.mean(Csig_p_6_in),np.mean(Csig_p_10_in)]
    Csig_out=[np.mean(Csig_m_10_out),np.mean(Csig_m_6_out),np.mean(Csig_m_2_out),np.mean(Csig_p_2_out),np.mean(Csig_p_6_out),np.mean(Csig_p_10_out)]
    Cbak_in=[np.mean(Cbak_m_10_in),np.mean(Cbak_m_6_in),np.mean(Cbak_m_2_in),np.mean(Cbak_p_2_in),np.mean(Cbak_p_6_in),np.mean(Cbak_p_10_in)]
    Cbak_out=[np.mean(Cbak_m_10_out),np.mean(Cbak_m_6_out),np.mean(Cbak_m_2_out),np.mean(Cbak_p_2_out),np.mean(Cbak_p_6_out),np.mean(Cbak_p_10_out)]
    Covl_in=[np.mean(Covl_m_10_in),np.mean(Covl_m_6_in),np.mean(Covl_m_2_in),np.mean(Covl_p_2_in),np.mean(Covl_p_6_in),np.mean(Covl_p_10_in)]
    Covl_out=[np.mean(Covl_m_10_out),np.mean(Covl_m_6_out),np.mean(Covl_m_2_out),np.mean(Covl_p_2_out),np.mean(Covl_p_6_out),np.mean(Covl_p_10_out)]
    
    return pesq_in,pesq_out,Csig_in,Csig_out,Cbak_in,Cbak_out,Covl_in,Covl_out,num_of_examples_per_SNR