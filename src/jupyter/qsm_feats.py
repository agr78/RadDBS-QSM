__author__ = 'Alexandra Roberts'

import torch
import torchvision
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import os
import numpy as np
from scipy.spatial import cKDTree
from datetime import datetime
from util import get_neighbors, l1, neighboring_features

class MLP(nn.Module):
    def __init__(self,in_size,kernel_size,cnn_layers,n_channels,fc_layers):
        super(MLP, self).__init__()
        torch.manual_seed(0)
        self.layers = nn.ModuleList()
        self.fc_layers = fc_layers
        self.n_channels = n_channels
        self.n_layers = cnn_layers
        self.in_size = in_size
        self.kernel_size = kernel_size
        # Convolutional layers
        for layer in np.arange(self.n_layers):
            if layer == 0:
                self.layers.append(nn.Conv3d(in_channels=1,out_channels=self.n_channels,kernel_size=self.kernel_size))
            else:
                self.layers.append(nn.Conv3d(in_channels=self.n_channels,out_channels=self.n_channels,kernel_size=self.kernel_size))
        self.out_size = (self.in_size[0],self.n_channels,
                        self.in_size[1]-(self.n_layers)*(self.kernel_size[0]-1),
                        self.in_size[2]-(self.n_layers)*(self.kernel_size[1]-1),
                        self.in_size[3]-(self.n_layers)*(self.kernel_size[2]-1))
    
        # Fully-connected layers
        for layer in np.arange(self.fc_layers):
            self.layers.append(nn.Linear(self.out_size[1]*self.out_size[2]*self.out_size[3]*self.out_size[4],
                                         self.out_size[1]*self.out_size[2]*self.out_size[3]*self.out_size[4]))
        # Regression layer
        self.layers.append(nn.Linear(self.out_size[1]*self.out_size[2]*self.out_size[3]*self.out_size[4],1))
 
    def encode(self, x):
        # Forward pass of convolutional layers
        for layer in self.layers[:-(self.fc_layers+1)]:
            x = F.relu(layer(x))
        # Forward pass of fully-connected layers
        fc_count = 0
        for layer in self.layers[-self.fc_layers:-1]:
            idz = -self.fc_layers+fc_count
            z = self.layers[idz](x.reshape(x.shape[0],x.shape[1]*x.shape[2]*x.shape[3]*x.shape[4]))
            x = z.reshape(x.shape)
            fc_count = fc_count+1
        # Forward pass of regression layer
        z = self.layers[-1](x.reshape(x.shape[0],x.shape[1]*x.shape[2]*x.shape[3]*x.shape[4]))
        return z

    def forward(self, x):
        y = self.encode(x)
        return y
    

def train_model(X_all,y_all,model,X_test,solver,
                lr,lr_decay,alpha,reg_type,
                num_epochs,batch_size,num_neighbors,random_val,
                early_stopping, verbose,save_state,case_id):
    val_synth = 0
    train_curve = np.zeros((num_epochs,1))
    val_curve = np.zeros((num_epochs,1))
    if num_neighbors != 0:
        if val_synth == 1:
            ws,n_vals = cKDTree(X_all).query(X_test, k=num_neighbors)
            W = np.sum(ws)
            X_val = 0
            y_val = 0
            for j in np.arange(len(ws[0])):
                X_val += (ws[0][j]/W)*X_all[n_vals[0][j],:]
                y_val += (ws[0][j]/W)*y_all[n_vals[0][j]]     
            y_val = torch.Tensor([y_val]).cuda()
            X_val = torch.Tensor(X_val).cuda()
            y_train = y_all
            X_train = X_all
        else:
            if random_val == True:
                np.random.seed(0)
                n_vals = np.random.choice(np.arange(len(y_all)),num_neighbors-3,replace=False)
                n_vals = np.concatenate((n_vals,np.expand_dims(np.argmax(y_all),axis=-1),
                                         np.expand_dims(np.where(y_all==np.median(y_all))[0][0],axis=-1),
                                         np.expand_dims(np.argmin(y_all),axis=-1)))
                print(np.where(y_all==np.median(y_all))[0][0])
            else:
                n_vals = cKDTree(X_all).query(X_test, k=num_neighbors)[1]
            X_val = torch.Tensor(X_all[n_vals, :]).cuda()
            y_val = torch.Tensor(y_all[n_vals].T).cuda()
            y_train = torch.Tensor(np.delete(y_all,n_vals,axis=0)).cuda()
            X_train = torch.Tensor(np.delete(X_all,n_vals,axis=0)).cuda()

        if reg_type == 'latent_dist':
            idy_all = np.squeeze(get_neighbors(y_all.reshape(-1, 1)))
            idy = np.delete(idy_all,n_vals,axis=0)
            idy_val = idy_all[n_vals]
            if np.ndim(X_all) == 2:
                Xn_val = neighboring_features(X_all,idy_val)
            else:
                Xn_val = neighboring_features(X_all.reshape(X_all.shape[0],np.prod(X_all.shape[1:])),idy_val)
                Xn_val = Xn_val.reshape(len(idy_val),X_all.shape[1],X_all.shape[2],X_all.shape[3])
    else:
        X_train = torch.Tensor(X_all).cuda()
        y_train = torch.Tensor(y_all).cuda()
        X_val = []
        y_val = []

    n_cases = X_train.shape[0]
    best_loss = np.Inf
    X_all = torch.Tensor(X_all).cuda()
    y_all = torch.Tensor(y_all).cuda()

    if verbose == True and reg_type == 'latent_dist' and num_neighbors != 0:
        print('Validation labels',str(y_val),'have nearest neighbors',str(y_all[idy_val]))
    torch.autograd.set_detect_anomaly(True)
    model.cuda()
    l2 = nn.MSELoss(reduction='sum')
    if solver == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay, gamma=0.1)
    num_batches = int(n_cases/batch_size)
    cum_loss = 0
    if verbose == True:
        print('Creating',str(num_batches),'batchs of size',str(batch_size),'from',str(n_cases),'training cases')
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch_case in np.arange(num_batches):
            # Get batch
            X_batch = X_train[int(batch_size*batch_case):int(batch_size*(batch_case+1)),:]
            y_batch = y_train[int(batch_size*batch_case):int(batch_size*(batch_case+1))]
      
            optimizer.zero_grad()
            yh = model(torch.unsqueeze(X_batch,dim=1))
            yh = yh.cuda()

            if reg_type == 'l1':
                L_params = [x.view(-1) for x in model.parameters()]
                weights = L_params[-2]
                l1_reg = l1(weights,0)
                l1_reg_val = l1_reg
            elif reg_type == 'latent_dist':
                idy_batch = idy[int(batch_size*batch_case):int(batch_size*(batch_case+1))]
                if np.ndim(X_all) == 2:
                    # Compute neighbors over batch or whole training set?
                    Xn_batch = neighboring_features(X_all.cpu(),idy_batch)
                else:
                    Xn_batch = neighboring_features(X_all.cpu().reshape(X_all.shape[0],np.prod(X_all.shape[1:])),idy_batch)
                    Xn_batch = Xn_batch.reshape(len(idy_batch),1,X_all.shape[1],X_all.shape[2],X_all.shape[3])  
                l1_reg = l1(model.encode(X_batch),model.encode(Xn_batch.cuda()))
                l1_reg_val = l1(model.encode(X_val),model.encode(Xn_val.cuda()))
            else:
                l1_reg = 0
                l1_reg_val = 0
                alpha = 0
                
            loss = (l2(torch.squeeze(yh),y_batch)+alpha*l1_reg)/len(y_batch)
            train_curve[epoch] = loss.detach().cpu()
            loss.backward()
            train_loss += loss
            cum_loss += loss
    
            if num_neighbors != 0:
                model.eval()
                yh_val = model(torch.unsqueeze(X_val,dim=1))
                val_loss = (l2(torch.squeeze(yh_val),y_val)+alpha*l1_reg_val)/len(y_val)
                val_curve[epoch] = val_loss.detach().cpu()
                # Save best validation model
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_epoch = epoch
                    if verbose == True:
                        print('Best validation loss:',str(val_loss.item()),
                            'at learning rate',str(optimizer.param_groups[0]['lr']),
                            'and epoch',epoch)
                    model_path = './trained_models/model_{}.pth'.format(case_id)
                    torch.save(model.state_dict(), model_path)

                if early_stopping==True:
                    res = abs(val_curve[epoch]-val_curve[epoch-1])/val_curve[epoch-1]
                    if res < 0.01:
                        print('Stopping at residual',str(res),'at epoch',str(epoch))
                        epoch = num_epochs
            # Save last epoch
            else:
                    best_loss = np.min(train_curve)
                    best_epoch = np.argmin(train_curve)
                    if epoch == num_epochs-1:
                        model_path = './trained_models/nn0_model_{}.pth'.format(case_id)
                        torch.save(model.state_dict(), model_path)

            optimizer.step()
           
            if verbose == True:
                if epoch % 10 == 0:
                    if reg_type == 'latent_dist':
                        print('Training batch labels',str(y_batch),'have neighbors',str(y_all[idy_batch]))
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} at learning rate {:.6f}'.format(
                    epoch,
                    int(batch_case+1),
                    int(num_batches), 100.*(batch_case/num_batches),
                    loss,
                    optimizer.param_groups[0]['lr']))

        if lr_decay != None:
            scheduler.step()

    y_val_list = np.asarray(['%.2f' % j for j in y_val],dtype=float)
    print('====> Epoch: {} Average loss: {:.4f} Best validation loss: {:.4f} at epoch: {}'.
          format(epoch+1,
          cum_loss/num_epochs,
          best_loss,best_epoch+1))
    print('Using neighbors',y_val_list,'for validation')
    model.load_state_dict(torch.load(model_path))
    yh = model(torch.unsqueeze(torch.Tensor(X_test),dim=1).cuda())

    if save_state == False:
        os.remove(model_path)
    return yh, model, X_train, y_train, X_val, y_val, train_curve, val_curve