#%%
import os
import torch
import hydra
from config import massagePassingConfig 
from hydra.core.config_store import ConfigStore
from torch_geometric.data import DataLoader as DataLoader_G
import time
from torch.utils.tensorboard import SummaryWriter
from data.DataLoader import MyGraphDataset
from train import train
from train import val
from test_batchs_ref import test as test_ref
from test_save_wavs import test as test_save_wavs
from DNS.test_DNSMOS import test_DNSMOS
print("finished import ")
print("PyTorch has version {}".format(torch.__version__))
print("cuda is available? {}" .format(torch.cuda.is_available()))
import shutil
import torch.optim as optim

from model.massage_passing import *

cs = ConfigStore.instance()
cs.store(name="massagePassingConfig", node=massagePassingConfig)
@hydra.main(config_path = "conf", config_name = "config")

def main(cfg: massagePassingConfig): 

    # save files as part of hydra output    
    shutil.copy(hydra.utils.get_original_cwd() + '/main.py', 'save_main.py')
    shutil.copy(hydra.utils.get_original_cwd() + '/model/massage_passing.py', 'save_massage_passing.py')
    shutil.copy(hydra.utils.get_original_cwd() + '/train.py', 'save_train.py')
    shutil.copy(hydra.utils.get_original_cwd() + '/test_batchs_ref.py', 'save_test_ref.py')
    
    # print data set paths
    print("train_path: ",cfg.paths.train_path)
    print("test_path: ",cfg.paths.test_path)
    
    # load data set
    train_data = MyGraphDataset(cfg.paths.train_path)
    val_data = MyGraphDataset(cfg.paths.val_path)
    test_data = MyGraphDataset(cfg.paths.test_path)
    
    # train validation and test loaders 
    train_loader = DataLoader_G(train_data, batch_size= cfg.model_hp.batchSize, shuffle=True,num_workers=cfg.model_hp.num_workers)
    val_loader = DataLoader_G(val_data, batch_size= cfg.model_hp.batchSize, shuffle=False,num_workers=cfg.model_hp.num_workers)
    test_loader = DataLoader_G(test_data, batch_size= cfg.model_hp.batchSize_test, shuffle=False,num_workers=cfg.model_hp.num_workers)
    # Defining the model - graph or linear
    if cfg.model_hp.model_type == 'graph':
        model = NetMulti(cfg.modelParams)
    elif cfg.model_hp.model_type == 'linear':
        model = LinearMultiNet(cfg.modelParams)
    # print model info and number of parameters    
    print(model)
    print("number of parameters: {}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    print("number of trainable parameters: {}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    print("number of non-trainable parameters: {}".format(sum(p.numel() for p in model.parameters() if not p.requires_grad)))
    print("number of layers: {}".format(len(list(model.parameters()))))
    print("number of trainable layers: {}".format(len(list(model.parameters()))))
    print("number of non-trainable layers: {}".format(len(list(model.parameters()))))
    # Defining the device 
    device_ids = [cfg.device.device_num]
    device = torch.device(f"cuda:{device_ids[0]}" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)
    
    # Defining the optimizer
    if cfg.optimizer.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=cfg.optimizer.learning_rate, weight_decay=cfg.optimizer.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
    else:
        print("error no optimzier has been configrues - got optimzier {}".format(cfg.optimizer.optimizer))
    
    # Training the model 
    train_loss = []
    val_loss = []
    # load weights if test flag is on 
    if cfg.flags.test:
        load_path = cfg.paths.modelData_path_saved + 'model_epoc_100.pt'
        model.load_state_dict(torch.load(load_path))
        print("load weights from: {}".format(load_path))
        test_save_wavs(model, cfg.params, cfg.paths.results_path, test_loader, device,cfg.flags)
        test_DNSMOS(cfg.paths.results_path,cfg.paths.csv_path,100)
        print('done test_save_wavs and test_DNSMOS')
        test_ref(model, cfg.params, cfg.paths.results_path, test_loader, device,100,cfg.flags)
        print('done test') 
    # train and test 
    else:
        writer = SummaryWriter(log_dir = cfg.paths.log_path)
        print('writer_path:',cfg.paths.log_path)
        ''' start_time = time.time()
        print("start test")   
        test_ref(model, cfg.params, cfg.paths.results_path, test_loader, device,0,cfg.flags)
        test_save_wavs(model, cfg.params, cfg.paths.results_path, test_loader, device,0)
        test_DNSMOS(cfg.paths.results_path,cfg.paths.csv_path,0)
        print("end test time:{} ",time.time()-start_time) '''
        for epoch in range(cfg.model_hp.epochs):
            epoch_train_loss = train(model, cfg.params, train_loader, optimizer, device, cfg.loss)
            epoch_val_loss = val(model, cfg.params,val_loader, device, cfg.loss,lr_scheduler) 
            train_loss.append(epoch_train_loss / len(train_loader)) 
            val_loss.append(epoch_val_loss[cfg.loss.loss_type] / len(val_loader))      
            if (epoch+1)%10==0:
                if cfg.flags.save_model:
                    if not os.path.exists(cfg.paths.modelData_path):
                        os.makedirs(cfg.paths.modelData_path)
                    torch.save(model.state_dict(),cfg.paths.modelData_path + 'model_epoc_' + str(epoch+1) +'.pt',_use_new_zipfile_serialization=False)
                       
                start_time = time.time()
                print("start test")   
                test_ref(model, cfg.params, cfg.paths.results_path, test_loader, device,epoch+1,cfg.flags)
                print("end test time:{} ",time.time()-start_time)
                test_save_wavs(model, cfg.params, cfg.paths.results_path, test_loader, device,epoch+1,cfg.flags)
                test_DNSMOS(cfg.paths.results_path,cfg.paths.csv_path,epoch+1)
            print(f"Epoch {epoch+1}/{cfg.model_hp.epochs}, train_loss: {train_loss[epoch]}, val_loss: {val_loss[epoch]}")
            # Saved results in log file (TensorBorad)
            writer.add_scalar("Loss/train "+cfg.loss.loss_type,epoch_train_loss / len(train_loader),epoch)
            writer.add_scalar("Loss/val L1",epoch_val_loss['L1'] / len(val_loader),epoch)
            writer.add_scalar("Loss/val L2",epoch_val_loss['L2'] / len(val_loader),epoch)
            writer.add_scalar("Loss/val SISDR 1",epoch_val_loss['si_sdr_1'] / len(val_loader),epoch) 
            writer.add_scalar("Loss/val SISDR 2",epoch_val_loss['si_sdr_2'] / len(val_loader),epoch)
            writer.add_scalar("Loss/val SISDR on ref",epoch_val_loss['si_sdr_on_ref'] / len(val_loader),epoch)
            writer.add_scalar("Loss/val NPM",epoch_val_loss['NPM'] / len(val_loader),epoch)
            writer.add_scalar("Loss/val SBF",epoch_val_loss['SBF'] / len(val_loader),epoch)
            writer.add_scalar("Loss/val Blocking_loss",epoch_val_loss['Blocking_loss'] / len(val_loader),epoch)
            writer.add_scalar("Loss/val Blocking_loss_with_noise",epoch_val_loss['Blocking_loss_with_n'] / len(val_loader),epoch)
            writer.add_scalar("Loss/val STOI",epoch_val_loss['STOI'] / len(val_loader),epoch)
            writer.add_scalar("Loss/val ESTOI",epoch_val_loss['ESTOI'] / len(val_loader),epoch)
            writer.add_scalar("Loss/val L1_RTFs",epoch_val_loss['L1_RTFs'] / len(val_loader),epoch)
        writer.flush()
        writer.close()
        


if __name__ == '__main__':
    print('start')
    main()
    
    