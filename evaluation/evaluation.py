#%%
import os
import torch
import hydra
from config import massagePassingConfig 
from hydra.core.config_store import ConfigStore
from torch_geometric.data import DataLoader as DataLoader_G
import sys
from create_data.create_example import create_example
from create_data.estimate_RTF import estimate_RTFs
from create_data.connect_RTFs_to_graphs import connect_RTFs_to_graphs
from test_and_evaluation_measurments import test
from utils_test import MyGraphDataset,print_DNSMOS_results
sys.path.append("../")
from DNS.test_DNSMOS import test_DNSMOS
from model.massage_passing import *

cs = ConfigStore.instance()
cs.store(name="massagePassingConfig", node=massagePassingConfig)
@hydra.main(config_path = "conf", config_name = "config_evaluation")

def main(cfg: massagePassingConfig): 


    # create new example and estimate RTFs 
    #create_example(cfg.paths,cfg.params)
    #estimate_RTFs(cfg.paths,cfg.params)
    # connect RTFs to graph by KNN 
    #connect_RTFs_to_graphs(cfg.paths,cfg.params)
        
    # load data set
    test_data = MyGraphDataset(cfg.paths.save_graphs_path)
    # train validation and test loaders 
    test_loader = DataLoader_G(test_data, batch_size= cfg.model_hp.batchSize_test, shuffle=False,num_workers=cfg.model_hp.num_workers)
    # Defining the model - graph or linear
    
    model = NetMulti(cfg.modelParams)
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

    # load weights if test flag is on 
    model.load_state_dict(torch.load(cfg.paths.modelData_path_saved))
    print("load weights from: {}".format(cfg.paths.modelData_path_saved))
    test(model, cfg.params, cfg.paths, test_loader, device)
    test_DNSMOS(cfg.paths.save_wav_dir,cfg.paths.csv_path,100)
    print_DNSMOS_results(cfg.paths.csv_path)
    print("finished")


if __name__ == '__main__':
    print('start')
    main()
    
    