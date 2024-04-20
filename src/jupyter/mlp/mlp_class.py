__author__ = 'Alexandra Roberts'

import sys
sys.path.append('../')
import torch
from torch import nn, optim
import copy
from torch.utils.data import DataLoader
from pytorch_adapt.layers import coral_loss
from torchvision.ops import sigmoid_focal_loss as FocalLoss
import os
import numpy as np
from scipy.stats import linregress
from scipy.spatial import cKDTree
from sklearn.model_selection import train_test_split
from util import get_neighbors, l1, neighboring_features
from joblib import Parallel, delayed
from datasets import QSM_slices
import cProfile, pstats, io
from pstats import SortKey
import gc
import glob
import matplotlib.pyplot as plt

# Matrix product Aw for Lasso cost function
class Net(nn.Module):
    def __init__(self,in_size,n_channels,ks,mp,rad_dim):
        super(Net, self).__init__()
        torch.manual_seed(0)
        self.in_size = in_size
        self.n_channels = n_channels
        self.ks = ks
        self.rad_dim = rad_dim
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=self.n_channels, kernel_size=self.ks, stride=1)
        self.relu = nn.ReLU()
        self.dp = nn.Dropout(p=0.05,inplace=True)
        self.conv2 = nn.Conv2d(in_channels=self.n_channels, out_channels=self.n_channels, kernel_size=self.ks, stride=1) 
        self.sm = nn.Sigmoid()
        n_layers = 2
        print('Using constant kernel and channel sizes')

        if mp == True:
            self.out_size = (self.n_channels, 24, 24)# (self.in_size-(n_layers)*(self.ks-1))//self.mp.kernel_size,
                        # (self.in_size-(n_layers)*(self.ks-1))//self.mp.kernel_size)
        else:
            self.out_size = (self.n_channels, self.in_size-(n_layers)*(self.ks-1),
                         self.in_size-(n_layers)*(self.ks-1))
      
        self.latent_size = (self.n_channels,self.out_size[1],self.out_size[2])
        self.fc1 = nn.Linear(np.prod(self.out_size),np.prod(self.out_size))
        self.r = nn.Linear(np.prod(self.out_size),1)
    # Estimate Aw = b
    def encode(self, x):
        zc1 = self.relu(self.conv1(x))
        zc2 = self.relu(self.conv2(zc1))
        zc5 = zc2
        zfc = self.relu(self.fc1(zc5.view(zc5.shape[0], -1)))
        z = zfc
        #z = self.dp(torch.concat((self.relu(self.fc2(zfc)),zfc),dim=1))
        return z
    # Forward pass
    def forward(self, x, P):
        z = self.encode(x)
        # Append radiomics as skip connection
        if self.rad_dim != 0:
            y = self.sm(self.r(torch.concat((z.view(z.shape[0],-1),P),dim=1)))
        else:
            y = self.sm(self.dp(self.r(z.view(z.shape[0],-1))))
        return y

def train(model,train_loader,reg_type,optimizer):
    l2 = FocalLoss(reduction='mean')
    N = len(train_loader.dataset)
    for idx, (inputs, X, targets) in train_loader:
        y_pred = model(inputs,X)
        loss = (1/(2*N))*l2(y_pred,targets.cuda())
    return loss

def validate(val_loader,model,reg_type,alpha_f,gamma_f):
    model.eval()
    loss = 0
    with torch.no_grad():
        for idx, (inputs, X, targets) in enumerate(val_loader):
            inputs, X, targets = inputs.cuda(), X.cuda(), targets.cuda()
            outputs = model(inputs,X)
            case_loss = FocalLoss(torch.squeeze(outputs), torch.squeeze(targets),reduction='mean',alpha=alpha_f,gamma=gamma_f)
            loss = loss+case_loss
    return loss, outputs, targets, inputs

# Train model to find w*
def main(model,subsc,data_dir,subfolder,y,X,train_id,test_id,workers,num_epochs,lr,num_neighbors,batch_size,reg_type,alpha_f,gamma_f,aug_state,factor,save_state,verbose):
    data_dir_train = data_dir.copy()
    if aug_state == False:
        test_file = 'case_'+str(subsc[test_id][0])+'.npy'
        data_dir_train.remove(test_file)
        test_dataset = QSM_slices(data_dir=[test_file], subfolder=subfolder, aug_state=aug_state, factor=factor, X=X[test_id,:], subsc=subsc[test_id], targets=y[test_id], prefix=None)
        model_path = './trained_models/model_{}.pth'.format(subsc[test_id][0])
        train_dataset = QSM_slices(data_dir=data_dir_train,subfolder=subfolder, aug_state=aug_state, factor=factor, X=X[train_id,:],subsc=subsc[train_id],targets=y[train_id], prefix=None)
    else:
        print('Test ID',np.asarray(subsc)[test_id])
        cf = np.asarray(subsc)[test_id][0][0]
        print(cf)
        test_file = glob.glob('./tensor_slices_'+str(factor)+'/case_'+str(cf)+'*.npy')
        test_file = list(map(lambda x: x.replace('./tensor_slices_'+str(factor)+'/',''),test_file))
        print('Test file:',test_file)
        for file in test_file:
            data_dir_train.remove(file)
        test_file_out = sorted(test_file)[0]
        print('Creating test dataset with',test_file_out)
        test_dataset = QSM_slices(data_dir=[test_file_out], subfolder=subfolder, aug_state=aug_state, factor=factor, X=X[test_id,:], subsc=np.asarray(subsc)[test_id], targets=np.asarray(y)[test_id], prefix=None)
        model_path = './trained_models/model_{}.pth'.format(np.asarray(subsc)[test_id])
        train_dataset = QSM_slices(data_dir=data_dir_train,subfolder=subfolder,aug_state=aug_state, factor=factor, X=X[train_id,:],subsc=np.asarray(subsc)[train_id],targets=np.asarray(y)[train_id], prefix=None)
   
    N = len(train_dataset)
    num_batches = int(N/batch_size)
    train_loader = DataLoader(train_dataset, batch_size=N, shuffle=True,
                              num_workers=workers, pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=int(np.sum(test_id)), shuffle=False,
                             num_workers=workers, pin_memory=True, drop_last=False)
    model = torch.nn.DataParallel(model).cuda()

    # Data loss function term
    torch.autograd.set_detect_anomaly(True)
    best_loss = np.Inf
    best_model = model

    # Define optimizer
    if reg_type == 'l2':
        optimizer = optim.Adam(model.parameters(), weight_decay=alpha,amsgrad=True)
        #optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=alpha, momentum=0.9)
    else:
        optimizer = optim.Adam(model.parameters(),amsgrad=True) 
        #optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    
    # Model details
    if verbose > 1:
        print('Using '+str(torch.get_num_threads())+' threads')
        print('Creating',str(num_batches),'batchs of size',str(batch_size),'from',str(N),'training cases')
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print('Model contains',str(params),'trainable parameters')
        print(model)

    # Training loop
    pr = cProfile.Profile()
    if verbose > 2:
        pr.enable()
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        for idx, (inputs, X, targets) in enumerate(train_loader):
            y_pred = model(inputs,X)
            loss = FocalLoss(torch.squeeze(y_pred),torch.squeeze(targets.cuda()),reduction='mean',alpha=alpha_f,gamma=gamma_f)
            loss.backward()
            print(loss)
        if loss < best_loss:
            best_loss = loss
            best_model = copy.deepcopy(model)
            if verbose > 0:
                print('Best loss is: '+str(best_loss.item())+' at epoch: '+str(epoch))

        optimizer.step()
        #scheduler.step()

    torch.save(best_model.state_dict(), model_path)
    loss, y_pred, label, Xr = validate(train_loader,best_model,reg_type,alpha_f,gamma_f)
    print('Best model predicts',str(y_pred.mT > 0.3),'for',str(label.flatten() > 0),'with loss',FocalLoss(torch.squeeze(y_pred),torch.squeeze(label),reduction='mean',alpha=alpha_f,gamma=gamma_f),'and learning rate',optimizer.param_groups[0]['lr'])
    
    # Load best epoch and predict test case
    gc.collect()
    model.load_state_dict(torch.load(model_path))
    loss, yh, label, Xt = validate(test_loader,model,reg_type,alpha_f,gamma_f)
    if verbose > 2:
        pr.disable()
        s = io.StringIO()
        sortby = SortKey.CUMULATIVE
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())

    if save_state == False:
        os.remove(model_path)
    
    return yh, model, Xt
