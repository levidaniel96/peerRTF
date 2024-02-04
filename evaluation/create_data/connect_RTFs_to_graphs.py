#%%
import os
import numpy as np
from torch_geometric.nn import knn_graph
from torch_geometric.utils import to_undirected
from mat4py import loadmat
from torch.utils.data import Dataset
import torch
from torch_geometric.data import Data

train_size=3500
num_nodes=train_size+1

#%% Data_laoder 

class MyDataset2(Dataset):
  def __init__(self,path_first_spk,RTF_matrix_train_tensor,params,path_noisy=None):
    self.path_first_spk=path_first_spk
    self.path_noisy_data=path_noisy
      
    self.RTF_matrix_train_tensor=RTF_matrix_train_tensor
    self.params=params
  def getitem(self):
    new_path= self.path_first_spk+'RTFs_to_net.mat'   
    data = loadmat(new_path)
    RTFs=data['RTFs_to_net']
    matrix_RTFs=np.zeros(((self.params.M-1)*(self.params.Nl+self.params.Nr)))
    matrix_RTFs[:self.params.Nr]=RTFs[:self.params.Nr]
    matrix_RTFs[self.params.Nl+self.params.Nr-self.params.Nl:self.params.Nl+self.params.Nr]=RTFs[self.params.Nr:self.params.Nr+self.params.Nl]
    
    matrix_RTFs[(self.params.Nl+self.params.Nr):(self.params.Nl+self.params.Nr)+self.params.Nr]=RTFs[self.params.Nl+self.params.Nr:(self.params.Nl+self.params.Nr)+self.params.Nr]
    matrix_RTFs[2*(self.params.Nl+self.params.Nr)-self.params.Nl:2*(self.params.Nl+self.params.Nr)]=RTFs[self.params.Nl+self.params.Nr+self.params.Nr:(self.params.Nl+self.params.Nr)+self.params.Nl+self.params.Nr]
    
    matrix_RTFs[2*(self.params.Nl+self.params.Nr):2*(self.params.Nl+self.params.Nr)+self.params.Nr]=RTFs[2*(self.params.Nl+self.params.Nr):2*(self.params.Nl+self.params.Nr)+self.params.Nr]
    matrix_RTFs[3*(self.params.Nl+self.params.Nr)-self.params.Nl:3*(self.params.Nl+self.params.Nr)]=RTFs[2*(self.params.Nl+self.params.Nr)+self.params.Nr:2*(self.params.Nl+self.params.Nr)+self.params.Nl+self.params.Nr]
        
    matrix_RTFs[3*(self.params.Nl+self.params.Nr):3*(self.params.Nl+self.params.Nr)+self.params.Nr]=RTFs[3*(self.params.Nl+self.params.Nr):3*(self.params.Nl+self.params.Nr)+self.params.Nr]
    matrix_RTFs[4*(self.params.Nl+self.params.Nr)-self.params.Nl:4*(self.params.Nl+self.params.Nr)]=RTFs[3*(self.params.Nl+self.params.Nr)+self.params.Nr:3*(self.params.Nl+self.params.Nr)+self.params.Nl+self.params.Nr]
         
    self.noisy_h_first_spk=torch.unsqueeze(torch.tensor(matrix_RTFs,dtype=torch.float32 ),0)  

    new_path_noisy_data= self.path_noisy_data+'test_data.mat'
    data = loadmat(new_path_noisy_data)
    data_noisy=torch.tensor(data['y']) 
    data_ref_first_spk=torch.tensor(data['x'])[:,self.params.ref_mic]
    data_ref_first_spk_M=torch.tensor(data['x'])
    data_noise_M= torch.tensor(data['n'])
    data_SNR_in=torch.tensor(data['SNR_in'])
    self.RTF_matrix_train_tensor_to_net=torch.cat([self.RTF_matrix_train_tensor,self.noisy_h_first_spk])

      
    self.edge_index = knn_graph(self.RTF_matrix_train_tensor_to_net, 5, None, False)
    self.RTF_matrix_train_tensor_to_net=self.RTF_matrix_train_tensor_to_net.clone().detach()
    self.edge_index = to_undirected(self.edge_index,num_nodes= num_nodes)
    self.edge_index_temp=torch.empty((2,1),dtype=torch.int64)

    for edge in range(self.edge_index.shape[1]):
      if self.edge_index[0,edge]==train_size:
          self.edge_index_temp= torch.cat((self.edge_index_temp,torch.unsqueeze(self.edge_index[:,edge],1)),1)
    self.edge_index_temp=self.edge_index_temp[:,1:]
    self.edge_index = to_undirected(self.edge_index_temp, num_nodes=num_nodes)    
    self.RTF_matrix_train_tensor_to_net=self.RTF_matrix_train_tensor_to_net.reshape(self.RTF_matrix_train_tensor_to_net.shape[0],4,self.RTF_matrix_train_tensor_to_net.shape[1]//4)
    data = Data(x=self.RTF_matrix_train_tensor_to_net, edge_index=self.edge_index)
    mask_t = torch.zeros((num_nodes,), dtype=torch.bool)
    mask_t[-1] = 1
    data.mask = mask_t
    data.noisy_data=data_noisy.clone().detach()
    data.data_ref_first_spk=data_ref_first_spk.clone().detach()
    data.data_ref_first_spk_M=data_ref_first_spk_M.clone().detach()
    data.data_noise_M= data_noise_M.clone().detach()
    data.data_SNR_in= data_SNR_in.clone().detach()
    return data

def connect_RTFs_to_graphs(paths,params):

    RTF_matrix_train=torch.load(paths.RTFs_tensor_train_path)
    dataset=MyDataset2(paths.test_RTFs_path,RTF_matrix_train,params,paths.test_data_path)
    data=dataset.getitem()
    if not os.path.exists(paths.save_graphs_path):
        os.makedirs(paths.save_graphs_path)
    torch.save(data, paths.save_graphs_path+'graph_data.pt')


