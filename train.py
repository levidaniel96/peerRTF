
import torch
from utils.criterion import *


def train(model, params, train_loader, optimizer, device, loss):
    """
    Train the neural network model on the training set.

    Args:
        model (torch.nn.Module): The neural network model.
        params (Namespace): Parameters for the training process.
        train_loader (DataLoader): DataLoader for training data.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        device (torch.device): Device (GPU or CPU) on which to perform training.
        loss (LossType): Object specifying the loss function and its parameters.

    Returns:
        float: Total loss accumulated during training.
    """
    model.train()
    loss_all = 0

    # Iterate through batches in the training DataLoader
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()

        # Compute RTFs (Relative Transfer Functions) from the model
        if data.edge_index is None:
            RTFs_first_spk = model(data.first_spk_noisy)
        else:
            x, edge_index, _ = data.x, data.edge_index, data.batch
            output = model(x, edge_index)
            output = output[data.mask]
            RTFs_first_spk = torch.reshape(output, (data.num_graphs, params.M-1, int(eval(params.feature_size)))).to(device)

        RTFs_first_spk_oracle = torch.reshape(data.first_spk_clean, (data.num_graphs, params.M-1, int(eval(params.feature_size)))).to(device)

        # Prepare data for MVDR (Minimum Variance Distortionless Response) loss
        if loss.loss_mode == 'MVDR':
            noisy_data = torch.reshape(data.noisy_data, ((data.num_graphs, params.fs*4, 5))).to(device)
            if loss.loss_type in ['si_sdr_on_ref', 'STOI', 'ESTOI']:
                ref_first = torch.reshape(data.data_ref_first_spk_M, ((data.num_graphs, params.fs*4, 5)))
                ref_first = ref_first[:, :, 2].to(device)
            else:
                ref_first = None

        # Prepare data for RTFs (Relative Transfer Functions) loss
        if loss.loss_mode == 'RTFs':
            if loss.loss_type == 'SBF':
                ref_first = torch.reshape(data.data_ref_first_spk_M, ((data.num_graphs, params.fs*4, 5))).to(device)
                mics_first = None
                ref_first = ref_first[:, :, 2]
                mics_n = None
                ref_n = None
            elif loss.loss_type in ['Blocking_loss', 'Blocking_loss_with_n']:
                ref_first = torch.reshape(data.data_ref_first_spk_M, ((data.num_graphs, params.fs*4, 5))).to(device)
                mics_first = ref_first[:, :, [0, 1, 3, 4]]
                ref_first = ref_first[:, :, 2]
                if loss.loss_type == 'Blocking_loss_with_n':
                    n = torch.reshape(data.data_noise_M, ((data.num_graphs, params.fs*4, 5))).to(device)
                    mics_n = n[:, :, [0, 1, 3, 4]]
                    ref_n = n[:, :, 2]
                else:
                    mics_n = None
                    ref_n = None
            else:
                ref_first = None
                mics_first = None
                mics_n = None
                ref_n = None

        # Compute the loss based on the specified mode (MVDR or RTFs)
        if loss.loss_mode == 'MVDR':
            loss_out = MVDR_noisy_and_oracle_loss(noisy_data, RTFs_first_spk, RTFs_first_spk_oracle, params, loss.loss_type, device, data.num_graphs, 'train', ref_first)
        elif loss.loss_mode == 'RTFs':
            if loss.loss_RTFs_shape == 'one_RTF':
                loss_out = loss_RTFs(torch.squeeze(RTFs_first_spk), torch.squeeze(RTFs_first_spk_oracle), loss, ref_first, mics_first, ref_n, mics_n, device, params, 'train')

        # Backpropagate and update model parameters
        loss_out.backward()
        loss_all += loss_out.item()
        optimizer.step()

    return loss_all



def val(model, params, val_loader, device, loss,lr_scheduler=None):
    """
    Validate the neural network model on the validation set.

    Args:
        model (torch.nn.Module): The neural network model.
        params (Namespace): Parameters for the validation process.
        val_loader (DataLoader): DataLoader for validation data.
        device (torch.device): Device (GPU or CPU) on which to perform validation.
        loss (LossType): Object specifying the loss function and its parameters.

    Returns:
        dict: Dictionary containing various loss values on the validation set.
    """
    model.eval()

    # Initialize dictionary to store different loss values
    loss_all_val = {'L1': 0, 'L2': 0, 'si_sdr_1': 0, 'si_sdr_2': 0, 'NPM': 0, 'SBF': 0,
                    'Blocking_loss': 0, 'Blocking_loss_with_n': 0, 'si_sdr_on_ref': 0,
                    'STOI': 0, 'ESTOI': 0, 'L1_RTFs': 0}
    epoch_val_loss = 0
    for data in val_loader:
        with torch.no_grad():
            # Move data to the specified device (GPU or CPU)
            data = data.to(device)

            # Compute RTFs (Relative Transfer Functions) from the model
            if data.edge_index is None:
                RTFs_first_spk = model(data.first_spk_noisy)
            else:
                x, edge_index, _ = data.x, data.edge_index, data.batch
                output = model(x, edge_index)   
                output = output[data.mask]    
                RTFs_first_spk = torch.reshape(output, (data.num_graphs, params.M-1, int(eval(params.feature_size)))).to(device)
     
            RTFs_first_spk_oracle = torch.reshape(data.first_spk_clean, (data.num_graphs, params.M-1, int(eval(params.feature_size)))).to(device)

            # Prepare data for MVDR (Minimum Variance Distortionless Response) loss
            noisy_data = torch.reshape(data.noisy_data, ((data.num_graphs, params.fs*4, 5))).to(device)
            ref_first_ = torch.reshape(data.data_ref_first_spk_M, ((data.num_graphs, params.fs*4, 5))).to(device)
            mics_first = ref_first_[:, :, [0, 1, 3, 4]]
            ref_first = ref_first_[:, :, 2]
            n = torch.reshape(data.data_noise_M, ((data.num_graphs, params.fs*4, 5))).to(device)
            mics_n = n[:, :, [0, 1, 3, 4]]
            ref_n = n[:, :, 2]        

            # Compute various loss values
            L1_loss, L2_loss, si_sdr_1_loss, si_sdr_2_loss, si_sdr_on_ref, stoi_loss, estoi_loss = MVDR_noisy_and_oracle_loss(noisy_data, RTFs_first_spk, RTFs_first_spk_oracle, params, loss.loss_type, device , data.num_graphs, 'val', ref_first)
            loss_NPM_1, loss_SBF_1, loss_Blocking_1, loss_Blocking_with_n_1, L1_RTFs = loss_RTFs(torch.squeeze(RTFs_first_spk), torch.squeeze(RTFs_first_spk_oracle), loss, ref_first, mics_first, ref_n, mics_n, device, params, 'val')
            
            if loss.loss_type =='si_sdr_on_ref':
                epoch_val_loss += si_sdr_on_ref
            elif loss.loss_type=='si_sdr_2':
                epoch_val_loss += si_sdr_2_loss
            elif loss.loss_type == 'STOI':
                epoch_val_loss += stoi_loss
            elif loss.loss_type == 'ESTOI':
                epoch_val_loss += estoi_loss
            elif loss.loss_type == 'NPM':
                epoch_val_loss += loss_NPM_1
            elif loss.loss_type == 'SBF':
                epoch_val_loss += loss_SBF_1
            elif loss.loss_type == 'Blocking_loss': 
                epoch_val_loss += loss_Blocking_1
            elif loss.loss_type == 'Blocking_loss_with_n':
                epoch_val_loss += loss_Blocking_with_n_1
            elif loss.loss_type == 'L1':
                epoch_val_loss += L1_loss
            elif loss.loss_type == 'L2':
                epoch_val_loss += L2_loss
            else:
                raise ValueError('Invalid loss type.')
       
            # Update the dictionary with loss values
            loss_NPM = loss_NPM_1
            loss_SBF = loss_SBF_1
            loss_Blocking = loss_Blocking_1
            loss_Blocking_with_n = loss_Blocking_with_n_1
            loss_all_val['L1'] += L1_loss.item()
            loss_all_val['L2'] += L2_loss.item()
            loss_all_val['si_sdr_1'] += si_sdr_1_loss.item()
            loss_all_val['si_sdr_2'] += si_sdr_2_loss.item()
            loss_all_val['NPM'] += loss_NPM.item()
            loss_all_val['SBF'] += loss_SBF.item()
            loss_all_val['Blocking_loss'] += loss_Blocking.item()
            loss_all_val['Blocking_loss_with_n'] += loss_Blocking_with_n.item()  
            loss_all_val['L1_RTFs'] += L1_RTFs.item()
            loss_all_val['si_sdr_on_ref'] += si_sdr_on_ref.item()       
            loss_all_val['STOI'] += stoi_loss.item()
            loss_all_val['ESTOI'] += estoi_loss.item()   
            loss_all_val['L1_RTFs'] += L1_loss.item()
    if lr_scheduler:
        lr_scheduler.step(epoch_val_loss)  # Update the learning rate   
    return loss_all_val

   
