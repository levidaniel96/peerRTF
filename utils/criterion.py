
import torch
from torch import Tensor
from torch_stoi import NegSTOILoss


def NPM_loss(estimated_RTFs, oracle_RTFs):
    # Calculate the difference between oracle_RTFs and an estimate based on estimated_RTFs
    epsilon = oracle_RTFs - ((torch.diagonal((oracle_RTFs.conj() @ estimated_RTFs.T)) /
                             (torch.diagonal(estimated_RTFs.conj() @ estimated_RTFs.T) + 1e-20)) *
                            estimated_RTFs.conj().T).T   
    # Calculate the loss using a formula involving logarithms and norms
    loss = torch.mean(20 * torch.log10(torch.norm(epsilon, dim=1) / (torch.norm(oracle_RTFs, dim=1)) + 1e-20))   
    # Return the calculated loss
    return loss
 

def SBF_loss(estimated_RTFs, oracle_RTFs, ref):
    # Cross-correlation of reference signal 'ref' with flipped oracle_RTFs and estimated_RTFs
    r_n = torch.nn.functional.conv1d(ref.unsqueeze(1), torch.flip(oracle_RTFs, (0,)).view(1, 1, -1)).squeeze() - torch.nn.functional.conv1d(ref.unsqueeze(1), torch.flip(estimated_RTFs, (0,)).view(1, 1, -1)).squeeze()
    
    # Cross-correlation of reference signal 'ref' with flipped oracle_RTFs
    ref_filter = torch.nn.functional.conv1d(ref.unsqueeze(1), torch.flip(oracle_RTFs, (0,)).view(1, 1, -1)).squeeze()
    
    # signal blocking factor (SBF) calculation using logarithms and variances
    sbf = 10 * torch.log10(torch.var(ref_filter, dim=1) / (torch.var(r_n, dim=1) + 1e-20) + 1e-20)
    
    # Return the mean value of the calculated SBF across all dimensions
    return sbf.mean()

def Blocking_loss_with_n(estimated_RTFs, ref_s, mic_s, ref_n, mic_n, params):
    # Create a dirac tensor with ones at the specified Nl position
    dirac = torch.zeros(estimated_RTFs.shape[0], estimated_RTFs.shape[1]).to(estimated_RTFs.device)
    dirac[:, params.Nl] = 1
    
    # Calculate the difference for the signal part (s)
    s = torch.nn.functional.conv1d(mic_s.unsqueeze(1), torch.flip(dirac, (0,)).view(1, 1, -1)).squeeze() - torch.nn.functional.conv1d(ref_s.unsqueeze(1), torch.flip(torch.fft.ifftshift(estimated_RTFs), (0,)).view(1, 1, -1)).squeeze()
    
    # Calculate the difference for the noise part (n)
    n = torch.nn.functional.conv1d(mic_n.unsqueeze(1), torch.flip(dirac, (0,)).view(1, 1, -1)).squeeze() - torch.nn.functional.conv1d(ref_n.unsqueeze(1), torch.flip(torch.fft.ifftshift(estimated_RTFs), (0,)).view(1, 1, -1)).squeeze()

    # Blocking loss calculation using logarithms and variances
    blocking = 10 * torch.log10(torch.var(n, dim=1) / (torch.var(s, dim=1) + 1e-20) + 1e-20)
    
    # Return the mean value of the calculated blocking loss across all dimensions
    return blocking.mean()

def Blocking_loss(estimated_RTFs, ref_s, mic_s, params):
    # Use the microphone signal as 's'
    s = mic_s
    
    # Create a dirac tensor with ones at the specified Nl position
    dirac = torch.zeros(estimated_RTFs.shape[0], estimated_RTFs.shape[1]).to(estimated_RTFs.device)
    dirac[:, params.Nl] = 1
    
    # Calculate the difference for the noise part (r_n)
    r_n = torch.nn.functional.conv1d(mic_s.unsqueeze(1), torch.flip(dirac, (0,)).view(1, 1, -1)).squeeze() - torch.nn.functional.conv1d(ref_s.unsqueeze(1), torch.flip(torch.fft.ifftshift(estimated_RTFs), (0,)).view(1, 1, -1)).squeeze()

    # Blocking loss calculation using logarithms and variances
    blocking = 10 * torch.log10(torch.var(s, dim=1) / (torch.var(r_n, dim=1) + 1e-20) + 1e-20)
    
    # Return the mean value of the calculated blocking loss across all dimensions
    return blocking.mean()

def H_k_loss(estimated_RTFs, oracle_RTFs, device):
    # Compute the FFT of oracle and estimated RTFs
    H_org = torch.fft.fft(oracle_RTFs, dim=2)
    H_net = torch.fft.fft(estimated_RTFs, dim=2)
    
    # Get the batch size, number of microphones (M), and FFT size (nfft)
    batch, M, nfft = H_org.shape
    
    # Calculate the number of unique points (NUP) in the FFT
    NUP = nfft // 2 + 1
    
    # Set M to 4 (without reference)
    M = 4
    
    # Initialize the error tensor with complex values
    error = torch.zeros(batch, M, NUP, dtype=torch.complex64)
    
    # Create an identity matrix repeated for each batch
    eye_M = torch.eye(M).repeat(batch, 1, 1).to(device)
    
    # Loop over each frequency point (k)
    for k in range(NUP):
        # Compute the error using matrix operations
        error[:, :, k] = torch.bmm((eye_M - torch.bmm(H_org[:, :, k].unsqueeze(2), H_org[:, :, k].conj().unsqueeze(1)) /
                                   torch.bmm(H_org[:, :, k].conj().unsqueeze(2), H_org[:, :, k].unsqueeze(1))),
                                   H_net[:, :, k].unsqueeze(2)).squeeze()
    
    # Calculate the mean of the error tensor
    error_mean = torch.mean(error)
    
    # Return the mean value of the calculated error
    return error_mean


def loss_RTFs(estimated_RTFs, oracle_RTFs, loss, ref_s, mic_s, ref_n, mic_n, device, params, train_or_val_flag='train'):
    '''Calculate the loss based on the specified loss type on RTFs 
    Args:
        estimated_RTFs: Estimated RTFs
        oracle_RTFs: Oracle RTFs
        loss: Loss type 
        ref_s: Reference signal
        mic_s: Microphone signal
        ref_n: Reference noise
        mic_n: Microphone noise
        device: Device to use
        params: Parameters object
        train_or_val_flag: Flag to indicate if the loss is calculated for training or validation'''
    loss_NPM = 0
    loss_SBF = 0
    loss_Blocking = 0
    loss_Blocking_with_n = 0
    loss_L1 = 0
    
    # Loop over the microphones
    for m in range(estimated_RTFs.shape[1]):
        # Calculate losses based on the specified loss type
        if loss.loss_type == 'NPM' or train_or_val_flag == 'val':
            loss_NPM += NPM_loss(estimated_RTFs[:, m, :], oracle_RTFs[:, m, :])
            
        if loss.loss_type == 'SBF' or train_or_val_flag == 'val':
            loss_SBF -= SBF_loss(estimated_RTFs[:, m, :], oracle_RTFs[:, m, :], ref_s)
            
        if loss.loss_type == 'Blocking_loss' or train_or_val_flag == 'val':
            loss_Blocking -= Blocking_loss(estimated_RTFs[:, m, :], ref_s, mic_s[:, :, m], params)
            
        if loss.loss_type == 'Blocking_loss_with_n' or train_or_val_flag == 'val':
            loss_Blocking_with_n -= Blocking_loss_with_n(estimated_RTFs[:, m, :], ref_s, mic_s[:, :, m], ref_n, mic_n[:, :, m], params)
            
        if loss.loss_type == 'L1' or train_or_val_flag == 'val':
            loss_L1 += torch.nn.L1Loss()(estimated_RTFs[:, m, :], oracle_RTFs[:, m, :])
    
    # Return the total loss based on the specified loss type and training/validation flag
    if loss.loss_type == 'NPM' and train_or_val_flag == 'train':
        return loss.loss_scale * loss_NPM
    elif loss.loss_type == 'SBF' and train_or_val_flag == 'train':
        return loss.loss_scale * loss_SBF   
    elif loss.loss_type == 'Blocking_loss' and train_or_val_flag == 'train':
        return loss.loss_scale * loss_Blocking
    elif loss.loss_type == 'Blocking_loss_with_n' and train_or_val_flag == 'train':
        return loss.loss_scale * loss_Blocking_with_n
    elif loss.loss_type == 'L1' and train_or_val_flag == 'train':
        return loss.loss_scale * loss_L1
    elif train_or_val_flag == 'val':
        return loss.loss_scale * loss_NPM, loss.loss_scale * loss_SBF, loss.loss_scale * loss_Blocking, loss.loss_scale * loss_Blocking_with_n, loss.loss_scale * loss_L1



def _check_same_shape(preds: Tensor, target: Tensor) -> None:
    """Check that predictions and target have the same shape, else raise error."""
    if preds.shape != target.shape:
        raise RuntimeError(f"Predictions and targets are expected to have the same shape, pred has shape of {preds.shape} and target has shape of {target.shape}")


def scale_invariant_signal_distortion_ratio_loss(preds: Tensor, target: Tensor, zero_mean: bool = True) -> Tensor:
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
    
    return -val

def create_Qvv_k_batch(Y_STFT_matrix):
    '''
    calculate Qvv for each batch and each frequency point (k) in the STFT domain 
    Y_STFT_matrix: (batch_size,frame_count,M)
    '''  
    Rvv=torch.bmm(Y_STFT_matrix.permute(0, 2, 1),Y_STFT_matrix.conj())/len(Y_STFT_matrix[0,:,0])
    return Rvv

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

def ifft_shift_RTFs(RTFs, device, batch_size, M, wlen, Nr, Nl):
    # Calculate the total length of RTFs
    len_of_RTF = Nl + Nr
    
    # Initialize a tensor for shifted RTFs with zeros
    RTFs_ = torch.zeros((batch_size, wlen, M)).to(device)
    
    # Assign values to the shifted RTFs tensor based on indexing
    RTFs_[:, :Nr, 0] = RTFs[:, 0, len_of_RTF * 0:len_of_RTF * 0 + Nr]
    RTFs_[:, :Nr, 1] = RTFs[:, 1, len_of_RTF * 0:len_of_RTF * 0 + Nr]
    RTFs_[:, :Nr, 3] = RTFs[:, 2, len_of_RTF * 0:len_of_RTF * 0 + Nr]
    RTFs_[:, :Nr, 4] = RTFs[:, 3, len_of_RTF * 0:len_of_RTF * 0 + Nr]
    
    RTFs_[:, wlen - Nl:, 0] = RTFs[:, 0, len_of_RTF * 1 - Nl:len_of_RTF * 1]
    RTFs_[:, wlen - Nl:, 1] = RTFs[:, 1, len_of_RTF * 1 - Nl:len_of_RTF * 1]
    RTFs_[:, wlen - Nl:, 3] = RTFs[:, 2, len_of_RTF * 1 - Nl:len_of_RTF * 1]
    RTFs_[:, wlen - Nl:, 4] = RTFs[:, 3, len_of_RTF * 1 - Nl:len_of_RTF * 1]
    
    # Set the value at index (0, 2) to 1 for the reference microphone
    RTFs_[:, 0, 2] = 1
        
    return RTFs_



def MVDR_noisy_and_oracle_loss(y,RTFs_first_spk_net,RTFs_first_spk_oracle,args,loss_type,device,batch_size,train_or_val_flag='train',ref_s_first=None):
    '''
    calculate loss for MVDR algorhitem with noisy and oracle RTFs 
    y: (batch_size,frame_count,M)
    RTFs_first_spk_net: (batch_size,M,len_of_RTF)
    RTFs_first_spk_oracle: (batch_size,M,len_of_RTF)
    args: args object
    loss_type: loss type
    device: device to use
    batch_size: batch size
    train_or_val_flag: flag to indicate if the loss is calculated for training or validation
    ref_s_first: reference signal for the first speaker
    '''
    win = torch.hamming_window(args.wlen).to(device)
    e=1e-6
    eye_M=torch.eye(args.M).repeat(batch_size, 1, 1).to(device)
    frame_count = 1 + (y.shape[1] - args.wlen ) //args.n_hop
    if len(RTFs_first_spk_net.shape)==2:
        h_net_first_spk = torch.zeros((batch_size,args.wlen,args.M)).to(device)
        h_oracle_first_spk = torch.zeros((batch_size,args.wlen, args.M)).to(device)
        h_net_first_spk[:,:args.Nr, 0] = RTFs_first_spk_net[:,args.len_of_RTF * 0:args.len_of_RTF * 0 +args.Nr]
        h_net_first_spk[:,:args.Nr, 1] = RTFs_first_spk_net[:,args.len_of_RTF * 1:args.len_of_RTF * 1 +args.Nr]
        h_net_first_spk[:,:args.Nr, 3] = RTFs_first_spk_net[:,args.len_of_RTF * 2:args.len_of_RTF * 2 +args.Nr]
        h_net_first_spk[:,:args.Nr, 4] = RTFs_first_spk_net[:,args.len_of_RTF * 3:args.len_of_RTF * 3 +args.Nr]
        h_net_first_spk[:,args.NFFT-args.Nl:, 0] = RTFs_first_spk_net[:,args.len_of_RTF * 1-args.Nl:args.len_of_RTF * 1]
        h_net_first_spk[:,args.NFFT-args.Nl:, 1] = RTFs_first_spk_net[:,args.len_of_RTF * 2-args.Nl:args.len_of_RTF * 2]
        h_net_first_spk[:,args.NFFT-args.Nl:, 3] = RTFs_first_spk_net[:,args.len_of_RTF * 3-args.Nl:args.len_of_RTF * 3]
        h_net_first_spk[:,args.NFFT-args.Nl:, 4] = RTFs_first_spk_net[:,args.len_of_RTF * 4-args.Nl:args.len_of_RTF * 4]
        h_net_first_spk[:,0, 2] = 1
        
        h_oracle_first_spk[:,:args.Nr, 0] = RTFs_first_spk_oracle[:,args.len_of_RTF * 0:args.len_of_RTF * 0 +args.Nr]
        h_oracle_first_spk[:,:args.Nr, 1] = RTFs_first_spk_oracle[:,args.len_of_RTF * 1:args.len_of_RTF * 1 +args.Nr]
        h_oracle_first_spk[:,:args.Nr, 3] = RTFs_first_spk_oracle[:,args.len_of_RTF * 2:args.len_of_RTF * 2 +args.Nr]
        h_oracle_first_spk[:,:args.Nr, 4] = RTFs_first_spk_oracle[:,args.len_of_RTF * 3:args.len_of_RTF * 3 +args.Nr]
        h_oracle_first_spk[:,args.NFFT-args.Nl:, 0] = RTFs_first_spk_oracle[:,args.len_of_RTF * 1-args.Nl:args.len_of_RTF * 1]
        h_oracle_first_spk[:,args.NFFT-args.Nl:, 1] = RTFs_first_spk_oracle[:,args.len_of_RTF * 2-args.Nl:args.len_of_RTF * 2]
        h_oracle_first_spk[:,args.NFFT-args.Nl:, 3] = RTFs_first_spk_oracle[:,args.len_of_RTF * 3-args.Nl:args.len_of_RTF * 3]
        h_oracle_first_spk[:,args.NFFT-args.Nl:, 4] = RTFs_first_spk_oracle[:,args.len_of_RTF * 4-args.Nl:args.len_of_RTF * 4]
        h_oracle_first_spk[:,0, 2] = 1

    elif len(RTFs_first_spk_net.shape)==3:
        h_net_first_spk=ifft_shift_RTFs(RTFs_first_spk_net,device,batch_size,args.M,args.wlen,args.Nr,args.Nl)
        h_oracle_first_spk=ifft_shift_RTFs(RTFs_first_spk_oracle,device,batch_size,args.M,args.wlen,args.Nr,args.Nl)


    Y_STFT_matrix=torch.zeros((batch_size,int(eval(args.NUP)),frame_count,args.M),dtype=torch.cfloat).to(device)
    for m in range(args.M):
        Y_STFT_matrix[:,:,:,m]=torch.stft(y[:,:,m],args.NFFT,args.n_hop,args.wlen,win,center=False,return_complex=True)
    output_y_stft_net = torch.zeros(batch_size,int(eval(args.NUP)),frame_count, dtype=torch.cfloat).to(device)
    output_y_stft_oracle = torch.zeros(batch_size,int(eval(args.NUP)),frame_count, dtype=torch.cfloat).to(device)
  

    H_net_1 = torch.fft.fft(h_net_first_spk,dim=1)
    H_oracle_1 = torch.fft.fft(h_oracle_first_spk,dim=1)
    
    for f in range(int(eval(args.NUP))):
        # calculate Qvv for each batch and each frequency point (k) in the STFT domain 
        Qvv=create_Qvv_k_batch(Y_STFT_matrix[:,f,:frame_count//5,:])
        H_net_k = torch.unsqueeze(torch.squeeze(H_net_1[:,f,:]),2)
        H_oracle_k=torch.unsqueeze(torch.squeeze(H_oracle_1[:,f,:]),2)
        # MVDR weights calculation - w 
        inv_qvv = torch.inverse(Qvv+e*torch.norm(Qvv,dim=(1,2))[:, None, None]*eye_M) #+ e * LA.norm(Qvv[f, :, :]) * torch.eye(M).to(device))
        b_net = torch.bmm(inv_qvv,H_net_k)
        b_oracle = torch.bmm(inv_qvv, H_oracle_k)
        inv_temp_net = torch.squeeze(torch.bmm(H_net_k.conj().permute(0, 2, 1) , b_net)) + e*torch.norm(torch.bmm(H_net_k.conj().permute(0, 2, 1) , b_net),dim=(1,2))
        inv_temp_oracle= torch.squeeze(torch.bmm(H_oracle_k.conj().permute(0, 2, 1) , b_oracle)) + e*torch.norm(torch.bmm(H_oracle_k.conj().permute(0, 2, 1) , b_oracle),dim=(1,2))
        w_net =(torch.squeeze(b_net).T/inv_temp_net).T
        w_oracle = (torch.squeeze(b_oracle).T/inv_temp_oracle).T
        # calculate output for each batch and each frequency point (k) in the STFT domain
        output_y_stft_net[:,f,:]=torch.squeeze(torch.bmm(torch.unsqueeze(w_net.conj(),1) , torch.squeeze(Y_STFT_matrix[:,f,:,:]).permute(0, 2, 1)))
        output_y_stft_oracle[:,f,:]=torch.squeeze(torch.bmm(torch.unsqueeze(w_oracle.conj(),1) , torch.squeeze(Y_STFT_matrix[:,f,:,:]).permute(0, 2, 1)))
    y_hat_Net_0=torch.istft(output_y_stft_net,args.NFFT,args.n_hop,args.wlen,win) 
    y_hat_Oracle_0=torch.istft(output_y_stft_oracle,args.NFFT,args.n_hop,args.wlen,win)
    # calculate loss depending on the loss type and training/validation flag 
    if loss_type=='L1' or train_or_val_flag=='val':
        loss_f=torch.nn.L1Loss()
        L1_loss=1000*loss_f(y_hat_Oracle_0,y_hat_Net_0)
        if train_or_val_flag=='train':
            return L1_loss
    if loss_type=='L2' or train_or_val_flag=='val':
        loss_f=torch.nn.MSELoss()
        L2_loss=1000*loss_f(y_hat_Oracle_0,y_hat_Net_0)
        if train_or_val_flag=='train':
            return L2_loss
    if loss_type=='si_sdr_1' or train_or_val_flag=='val':
        si_sdr_1_loss=-10*si_sdr_torchaudio_calc(y_hat_Net_0,y_hat_Oracle_0)
        if train_or_val_flag=='train':
            return si_sdr_1_loss  
    if loss_type=='si_sdr_2' or train_or_val_flag=='val':  
        si_sdr_2_loss=10*torch.mean(scale_invariant_signal_distortion_ratio_loss(y_hat_Net_0,y_hat_Oracle_0))
        if train_or_val_flag=='train':
            return si_sdr_2_loss    
    if loss_type=='si_sdr_on_ref' or train_or_val_flag=='val':
        ref_s_first_STFT=torch.stft(ref_s_first,args.NFFT,args.n_hop,args.wlen,win,center=False,return_complex=True)
        ref_s_first_ISTFT=torch.istft(ref_s_first_STFT,args.NFFT,args.n_hop,args.wlen,win)
        si_sdr_on_ref_loss=-10*si_sdr_torchaudio_calc(y_hat_Net_0,ref_s_first_ISTFT)
        if train_or_val_flag=='train':
            return si_sdr_on_ref_loss
    if loss_type=='STOI' or train_or_val_flag=='val':
        stoi=NegSTOILoss(args.fs,use_vad=True,extended=False).to(device)
        ref_s_first_STFT=torch.stft(ref_s_first,args.NFFT,args.n_hop,args.wlen,win,center=False,return_complex=True)
        ref_s_first_ISTFT=torch.istft(ref_s_first_STFT,args.NFFT,args.n_hop,args.wlen,win)
        stoi_loss=stoi(y_hat_Net_0,ref_s_first_ISTFT).mean()       
        if train_or_val_flag=='train':
            return stoi_loss
    if loss_type=='ESTOI' or train_or_val_flag=='val':
        estoi=NegSTOILoss(args.fs,use_vad=True,extended=True).to(device)
        ref_s_first_STFT=torch.stft(ref_s_first,args.NFFT,args.n_hop,args.wlen,win,center=False,return_complex=True)
        ref_s_first_ISTFT=torch.istft(ref_s_first_STFT,args.NFFT,args.n_hop,args.wlen,win)
        estoi_loss=estoi(y_hat_Net_0,ref_s_first_ISTFT).mean()
        if train_or_val_flag=='train':
            return estoi_loss
    return L1_loss,L2_loss,si_sdr_1_loss,si_sdr_2_loss,si_sdr_on_ref_loss,stoi_loss,estoi_loss

