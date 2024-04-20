__author__ = 'Alexandra Roberts'

import sys
sys.path.append('../')
import torch
from torch import nn, optim 
from torch.utils.data import DataLoader
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
from fds import FDS

# Matrix product Aw for Lasso cost function
class Net(nn.Module):
    def __init__(self,in_size,n_channels,rad_dim,bucket_start):
        super(Net, self).__init__()
        torch.manual_seed(0)
        self.in_size = in_size
        self.n_channels = n_channels
        self.rad_dim = rad_dim
        self.bucket_start = bucket_start
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=self.n_channels, kernel_size=3, stride=1)
        self.mp = nn.MaxPool2d(kernel_size=self.conv1.kernel_size)
        self.lrn = nn.LocalResponseNorm(n_channels//4)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=self.n_channels, out_channels=self.n_channels, kernel_size=3, stride=1) 
        #self.mp2 = nn.MaxPool2d(kernel_size=self.conv2.kernel_size)
        self.conv3 = nn.Conv2d(in_channels=self.n_channels, out_channels=self.n_channels, kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(in_channels=self.n_channels, out_channels=self.n_channels, kernel_size=3, stride=1)
        self.conv5 = nn.Conv2d(in_channels=self.n_channels, out_channels=self.n_channels, kernel_size=3, stride=1)
        n_layers = 5
        print('Using constant kernel and channel sizes')
        self.out_size = (self.n_channels,14,14)
        #                  (self.in_size-(n_layers)*(self.conv5.kernel_size[0]-1),
        #                  self.in_size-(n_layers)*(self.conv5.kernel_size[0]-1),
        #                  self.n_channels)
        self.latent_size = (self.n_channels,self.out_size[1],2*self.out_size[2])
                        #(2*(self.in_size-(n_layers)*(self.conv5.kernel_size[0]-1)),
                         #2*(self.in_size-(n_layers)*(self.conv5.kernel_size[0]-1)),
                         #self.n_channels)
        self.fc1 = nn.Linear(np.prod(self.out_size),np.prod(self.out_size))
        self.fc2 = nn.Linear(np.prod(self.out_size),np.prod(self.out_size))
        self.FDS = FDS(
                feature_dim=2*np.prod(self.out_size)+self.rad_dim, bucket_num=10, bucket_start=self.bucket_start,
                start_update=0, start_smooth=1, kernel='gaussian', ks=9, sigma=1, momentum=0.9
            )
        self.start_smooth = self.FDS.start_smooth
        self.r = nn.Linear(np.prod(self.latent_size)+self.rad_dim,1)
    # Estimate Aw = b
    def encode(self, x):
        zc1 = self.relu(self.conv1(x))
        # print(zc1.shape)
        zc2 = self.lrn(self.mp(self.relu(self.conv2(zc1))))
        # print(zc2.shape)
        zc3 = self.relu(self.conv3(zc2))
        # print(zc3.shape)
        zc4 = self.relu(self.conv4(zc3))
        # print(zc4.shape)
        zc5 = self.relu(self.conv5(zc4))
        zfc = self.relu(self.fc1(zc5.view(zc5.shape[0], -1)))
        # Skip connection
        z = torch.concat((self.relu(self.fc2(zfc)),zfc),dim=1)
        return z
    # Forward pass
    def forward(self, x, P, epoch, y_train):
        z = self.encode(x)
        if epoch >= self.start_smooth:
            z = self.FDS.smooth(torch.concat((z.view(z.shape[0],-1),P),dim=1), y_train, epoch)
        else:
            z = torch.concat((z.view(z.shape[0],-1),P),dim=1)
        # Append UPDRS as skip connection
        y = self.r(z)
        return y, z

def train(model,train_loader,reg_type,optimizer):
    l2 = nn.MSELoss(reduction='sum')
    N = len(train_loader.dataset)
    for idx, (inputs, X, targets) in train_loader:
        
        y_pred = model(inputs,X)
        loss = (1/(2*N))*l2(y_pred,targets.cuda())

    return loss

def validate(val_loader,model,reg_type):
    l2 = nn.MSELoss()
    model.eval()
    loss = 0
    with torch.no_grad():
        for idx, (inputs, X, targets) in enumerate(val_loader):
            inputs, X, targets = inputs.cuda(), X.cuda(), targets.cuda()
            outputs, z = model(inputs,X,0,torch.zeros_like(targets))
            case_loss = l2(outputs, targets)
            loss = loss+case_loss
    return loss, outputs, targets

# Train model to find w*
def main(model,subsc,data_dir,y,X,train_id,test_id,workers,num_epochs,lr,num_neighbors,batch_size,reg_type,alpha,save_state,verbose):
    n_cases = int(np.sum(train_id))
    test_file = 'case_'+str(subsc[test_id][0])+'.npy'
    model_path = './trained_models_fds/model_{}.pth'.format(subsc[test_id][0])
    data_dir_train = data_dir.copy()
    data_dir_train.remove(test_file)
    train_dataset = QSM_slices(data_dir=data_dir_train, X=X[train_id,:], subsc=subsc[train_id], targets=y[train_id])
    test_dataset = QSM_slices(data_dir=[test_file], X=X[test_id,:], subsc=subsc[test_id], targets=y[test_id])
    train_loader = DataLoader(train_dataset, batch_size=n_cases, shuffle=True,
                              num_workers=workers, pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=int(np.sum(test_id)), shuffle=False,
                             num_workers=workers, pin_memory=True, drop_last=False)
    model = torch.nn.DataParallel(model).cuda()

    # Data loss function term
    torch.autograd.set_detect_anomaly(True)
    best_loss = np.Inf
    best_model = model
    N = len(train_loader.dataset)

    # Define optimizer
    if reg_type == 'l2':
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=alpha, momentum=0.9)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    num_batches = int(n_cases/batch_size)
    
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
        l2_loss = nn.MSELoss(reduction='sum')

        for idx, (inputs, X, targets) in enumerate(train_loader):
            y_pred, z = model(inputs,X,epoch,targets)
            loss = (1/(2*N))*l2_loss(y_pred,targets.cuda())
            loss.backward()
           
        if loss < best_loss:
            best_loss = loss
            best_model = model
            if verbose > 0:
                print('Best loss is: '+str(best_loss.item())+' at epoch: '+str(epoch))

        optimizer.step()

        if epoch > model.module.FDS.start_smooth:
            with torch.no_grad():
                encodings, labels = [], []
                for idx, (inputs, X, targets) in enumerate(train_loader):
                    inputs = inputs.cuda(non_blocking=True)
                    outputs, feature = model(inputs,X, epoch, targets)
                    encodings.extend(feature.data.squeeze().cpu().numpy())
                    labels.extend(targets.data.squeeze().cpu().numpy())
        
            encodings, labels = torch.from_numpy(np.vstack(encodings)).cuda(), torch.from_numpy(np.hstack(labels)).cuda()
            model.module.FDS.update_last_epoch_stats(epoch)
            model.module.FDS.update_running_stats(encodings, labels.ravel(), epoch)

        else:
            encodings = z
    torch.save(best_model.state_dict(), model_path)
    loss, y_pred, label = validate(train_loader,best_model,reg_type)
    print('Best model predicts',str(y_pred.mT),'for',str(label.mT))
    
    # Load best epoch and predict test case
    gc.collect()
    model.load_state_dict(torch.load(model_path))
    loss, yh, label = validate(test_loader,model,reg_type)
    if verbose > 2:
        pr.disable()
        s = io.StringIO()
        sortby = SortKey.CUMULATIVE
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())

    if save_state == False:
        os.remove(model_path)
    
    return yh, model, encodings
