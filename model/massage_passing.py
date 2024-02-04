
import torch
from torch_geometric.nn import  EdgeConv
import torch.nn as nn
import sys
sys.path.append("../")
from utils.criterion import *

def sigmoid_1(x):
    x=2*torch.sigmoid(x)-1
    return x

class Net(torch.nn.Module):
    def __init__(self,args):
        super(Net, self).__init__()
        self.args=args
        layers = []
        for i in range(args.num_layers):
            layers.append(nn.Linear(2*int(eval(args.feature_size)), 2*int(eval(args.feature_size))))
            layers.append(nn.Dropout(args.dropout))
            if args.batch_norm:
                layers.append(nn.BatchNorm1d(2*int(eval(args.feature_size))))
            if args.activation == 'tanh':
                layers.append(nn.Tanh())
            elif args.activation == 'relu':
                layers.append(nn.ReLU())
            elif args.activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            elif args.activation == 'LeakyReLU':
                layers.append(nn.LeakyReLU())
            elif args.activation == 'silu':
                layers.append(nn.SiLU())

            else:
                raise NotImplementedError  
        layers.append(nn.Linear(2*int(eval(args.feature_size)),int(eval(args.feature_size))))
        
        the_nn = nn.Sequential(*layers)
        self.conv1 = EdgeConv(the_nn, aggr="mean")
        #self.conv2 = EdgeConv(the_nn, aggr="mean")
    def forward(self, x, edge_index):

        x = self.conv1(x, edge_index)
        if self.args.activation_out == 'tanh':
            x=torch.tanh(self.args.tanh_alpha*x)
        elif self.args.activation_out == 'relu':
            x=torch.relu(x)
        elif self.args.activation_out == 'sigmoid':
            x=torch.sigmoid(x)  
        elif self.args.activation_out == 'LeakyReLU':
            layer=torch.nn.LeakyReLU()
            x=layer(x)
        elif self.args.activation_out == 'silu':
            layer=torch.nn.SiLU()
            x=layer(x)          
        elif self.args.activation_out == 'without':
            pass
        else:
            raise NotImplementedError
        return x

class NetMulti(torch.nn.Module):
    def __init__(self,args):
        super(NetMulti, self).__init__()
        self.args=args
        if self.args.train_mode== 'one_model':
            self.Net=Net(self.args)
        else:
            self.Net1=Net(self.args)
            self.Net2=Net(self.args)
            self.Net3=Net(self.args)
            self.Net4=Net(self.args)

        
    def forward(self, x, edge_index):
        if self.args.train_mode== 'one_model':
            x1=self.Net(x[:,0,:],edge_index).unsqueeze(1)
            x2=self.Net(x[:,1,:],edge_index).unsqueeze(1)
            x3=self.Net(x[:,2,:],edge_index).unsqueeze(1)
            x4=self.Net(x[:,3,:],edge_index).unsqueeze(1)
        else:
            x1=self.Net1(x[:,0,:],edge_index).unsqueeze(1)
            x2=self.Net2(x[:,1,:],edge_index).unsqueeze(1)
            x3=self.Net3(x[:,2,:],edge_index).unsqueeze(1)
            x4=self.Net4(x[:,3,:],edge_index).unsqueeze(1)
        x=torch.cat((x1,x2,x3,x4),dim=1)
        return x    
    
class LinearNet(torch.nn.Module):
    def __init__(self,args):
        super(LinearNet, self).__init__()
        self.args=args
        layers = []
        for i in range(args.num_layers):
            layers.append(nn.Linear(int(eval(args.feature_size)), int(eval(args.feature_size))))
            layers.append(nn.Dropout(args.dropout))
            if args.batch_norm:
                layers.append(nn.BatchNorm1d(int(eval(args.feature_size))))
            if args.activation == 'tanh':
                layers.append(nn.Tanh())
            elif args.activation == 'relu':
                layers.append(nn.ReLU())
            elif args.activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            elif args.activation == 'LeakyReLU':
                layers.append(nn.LeakyReLU())
            elif args.activation == 'silu':
                layers.append(nn.SiLU())

            else:
                raise NotImplementedError  
        layers.append(nn.Linear(int(eval(args.feature_size)),int(eval(args.feature_size))))
        
        self.the_nn = nn.Sequential(*layers)

    def forward(self, x):
        x = self.the_nn(x)
        if self.args.activation_out == 'tanh':
            x=torch.tanh(self.args.tanh_alpha*x)
        elif self.args.activation_out == 'relu':
            x=torch.relu(x)
        elif self.args.activation_out == 'sigmoid':
            x=torch.sigmoid(x)  
        elif self.args.activation_out == 'without':
            pass
        else:
            raise NotImplementedError
        return x
    
        
class LinearMultiNet(torch.nn.Module):
    def __init__(self,args):
        super(LinearMultiNet, self).__init__()
        self.args=args
        if self.args.train_mode== 'one_model':
            self.Net=LinearNet(self.args)
        else:
            self.Net1=LinearNet(self.args)
            self.Net2=LinearNet(self.args)
            self.Net3=LinearNet(self.args)
            self.Net4=LinearNet(self.args)    
            
    def forward(self, x):
        if self.args.train_mode== 'one_model':
            x1=self.Net(x[:,0,:]).unsqueeze(1)
            x2=self.Net(x[:,1,:]).unsqueeze(1)
            x3=self.Net(x[:,2,:]).unsqueeze(1)
            x4=self.Net(x[:,3,:]).unsqueeze(1)
        else:
            x1=self.Net1(x[:,0,:]).unsqueeze(1)
            x2=self.Net2(x[:,1,:]).unsqueeze(1)
            x3=self.Net3(x[:,2,:]).unsqueeze(1)
            x4=self.Net4(x[:,3,:]).unsqueeze(1)
        x=torch.cat((x1,x2,x3,x4),dim=1)    

        return x