__author__ = 'Alexandra Roberts'

import torch
from torch import nn, optim
import os
import numpy as np
from scipy.stats import linregress
from scipy.spatial import cKDTree
from sklearn.model_selection import train_test_split
from util import get_neighbors, l1, neighboring_features
from joblib import Parallel, delayed

# Matrix product Aw for Lasso cost function
class MLP(nn.Module):
    def __init__(self,in_size):
        super(MLP, self).__init__()
        torch.manual_seed(0)
        self.in_size = in_size
        self.r = nn.Linear(self.in_size, 1)
    # Estimate Aw = b
    def encode(self, x):
        z = self.r(x)
        return z
    # Forward pass
    def forward(self, x):
        y = self.encode(x)
        return y

# Train model to find w*
def train_model(X_all,y_all,model,X_test,warm_start_weights,early_stopping,tol,lr,lr_decay,alpha,reg_type,thresh,num_epochs,batch_size,num_neighbors,random_val,verbose,save_state,case_id):
    val_synth = 0
    yh_l = 0
    pcount = 0
    train_curve = np.zeros((num_epochs,1))
    val_curve = np.zeros((num_epochs,1))
    # Optional validation withholding
    if num_neighbors > 0:
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
                n_vals = np.random.choice(len(y_all)-1,num_neighbors,replace=False)
            else:
                n_vals = cKDTree(X_all).query(X_test, k=num_neighbors)[1]
            idy_all = np.squeeze(get_neighbors(y_all.reshape(-1, 1)))
            X_val = torch.Tensor(X_all[n_vals, :]).cuda()
            y_val = torch.Tensor(y_all[n_vals].T).cuda()
            y_train = torch.Tensor(np.delete(y_all,n_vals,axis=0)).cuda()
            X_train = torch.Tensor(np.delete(X_all,n_vals,axis=0)).cuda()
            idy = np.delete(idy_all,n_vals,axis=0)
            idy_val = idy_all[n_vals]
            X_all = torch.Tensor(X_all).cuda()
            y_all = torch.Tensor(y_all).cuda()
            Xn_val = neighboring_features(torch.squeeze(X_all).cpu(),idy_val)
            if verbose > 1:
                print('Validation labels',str(y_val),'have nearest neighbors',str(y_all[idy_val]))
    # Train on whole dataset
    else:
        X_train = torch.Tensor(X_all).cuda()
        y_train = torch.Tensor(y_all).cuda()
        X_val = []
        y_val = []
        n_cases = X_train.shape[0]
        best_loss = np.Inf

    # Data loss function term
    torch.autograd.set_detect_anomaly(True)
    model.cuda()
    if warm_start_weights != []:
        model.load_state_dict(torch.load(warm_start_weights), strict=False)
    l2 = nn.MSELoss(reduction='sum')

    # Define optimizer
    optimizer = optim.SGD(model.parameters(), lr=lr)
    step = 0
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay, gamma=0.1)
    num_batches = int(n_cases/batch_size)

    # Model details
    if verbose > 1:
        print('Creating',str(num_batches),'batchs of size',str(batch_size),'from',str(n_cases),'training cases')
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print('Model contains',str(params),'trainable parameters')
        print(model)
        if warm_start_weights != []:
            print('Warm start enabled')

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch_case in np.arange(num_batches):
            # Make batch
            X_batch = X_train[int(batch_size*batch_case):int(batch_size*(batch_case+1)),:]
            y_batch = y_train[int(batch_size*batch_case):int(batch_size*(batch_case+1))]
            N = len(y_batch)
            if reg_type == 'l1':
                L_params = [x.view(-1) for x in model.parameters()]
                # Regularize weights, including w0 bias
                weights = torch.hstack((L_params[-2],L_params[-1]))
                if thresh != None:
                    mask = weights[:-1]>thresh
                l1_reg = l1(weights,0)
                l1_reg_val = l1_reg
            elif reg_type == 'latent_dist':
                idy_batch = idy[int(batch_size*batch_case):int(batch_size*(batch_case+1))]
                # Compute neighbors over batch or whole training set?
                Xn_batch = neighboring_features(X_all.cpu(),idy_batch)
                l1_reg = l1(model.encode(X_batch),model.encode(Xn_batch.cuda()))
                l1_reg_val = l1(model.encode(X_val),model.encode(Xn_val.cuda()))
            else:
                l1_reg = 0
                l1_reg_val = 0

            # Evaluate
            optimizer.zero_grad()
            yh = model(X_batch)
            yh = yh.cuda()
            # Lasso loss function (or latent distance, depending on reg_type)
            loss = (1/(2*N))*l2(torch.squeeze(yh),y_batch)+alpha*l1_reg
            train_curve[epoch] = loss.detach().cpu()
            loss.backward()
            train_loss += loss

            # Optional validation loss
            if num_neighbors > 0:
                model.eval()
                yh_val = model(X_val)
                val_loss = (1/(2*N))*l2(torch.squeeze(yh_val),y_val)+alpha*l1_reg_val
                val_curve[epoch] = val_loss.detach().cpu()
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_epoch = epoch
                    if verbose > 1:
                        print('Best validation loss:',str(val_loss.item()),
                            'at learning rate',str(optimizer.param_groups[0]['lr']),
                            'and epoch',epoch)
                    model_path = './trained_models/model_{}.pth'.format(case_id)
                    torch.save(model.state_dict(), model_path)

            # Training loss 
            else:
                if loss < best_loss:
                    best_loss = loss
                    best_epoch = epoch
                    if verbose > 1:
                        print('Best training loss:',str(loss.item()),
                            'at learning rate',str(optimizer.param_groups[0]['lr']),
                            'and epoch',epoch)
                    model_path = './trained_models/model_{}_{}.pth'.format(case_id,'alpha')
                    warm_start_weights = model_path
                    torch.save(model.state_dict(), model_path)

            # Training progres
            optimizer.step()
            if verbose > 1:
                if epoch % 10 == 0:
                    if reg_type == 'latent_dist':
                        print('Training batch labels',str(y_batch),'have neighbors',str(y_all[idy_batch]))
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch,
                        int(batch_case+1),
                        int(num_batches), 100.*(batch_case/num_batches),
                        loss))
                    print(optimizer.param_groups[0]['lr'])

        # Learning rate decay
        if lr_decay != None:
            if num_neighbors > 0:
                scheduler.step(val_loss)
            else:
                #scheduler.step(loss)
                scheduler.step()
            if epoch % lr_decay == 0 and epoch > 0:
                step = step+1

        if num_neighbors > 0:
            y_val_list = np.asarray(['%.2f' % j for j in y_val],dtype=float)
            print('====> Epoch: {} Average loss: {:.4f} Best validation loss: {:.4f} at epoch: {}'.
                format(epoch+1,
                train_loss/(num_epochs*len(y_train[int(batch_size*batch_case):int(batch_size*(batch_case+1))])),
                best_loss,best_epoch+1))
            print('Using neighbors',y_val_list,'for validation')
        
        # Early stopping
        if early_stopping != []:
            dy = np.abs(yh.detach().cpu().numpy()-yh_l)/yh.detach().cpu().numpy()
            yh_l = yh.detach().cpu().numpy()
            if np.mean(dy) < tol:
                pcount = pcount+1
            if pcount == early_stopping:
                break
            
    # Load best epoch and predict test case
    model.load_state_dict(torch.load(model_path))
    model.eval()
    if verbose > 0:
        if early_stopping != []:
            print('Early stopping occuring at epoch',str(epoch))
        print('Predicted',str(yh.mT),'for',str(y_train))
        print('Testing on best loss:',str(best_loss.item()),'from epoch',str(best_epoch))
    yh = model(torch.Tensor(X_test).cuda())   
    
    if save_state == False and warm_start_weights == []:
        os.remove(model_path)
    return yh, model, [X_train, y_train, X_val, y_val, train_curve, val_curve]

def model_cv(X_all,y_all,model,X_test,warm_start_weights,early_stopping,tol,lrs,lr_decay,alphas,reg_type,thresh,num_epochs,batch_size,num_neighbors,random_val,verbose,save_state,case_id,cvn):
    err = np.zeros((cvn,len(alphas),len(lrs)))
    for i in np.arange(cvn):
        X_allt, X_testt, y_allt, y_testt = train_test_split(X_all,y_all,test_size=cvn/X_all.shape[0],random_state=i)
        for j in np.arange(len(alphas)):
            for k in np.arange(len(lrs)):
                if j == 0:
                    warm_start_weights = []
                else:
                    warm_start_weights = './trained_models/model_{}_{}.pth'.format(case_id,'alpha')
                yh, model, _ = train_model(
                    X_allt,y_allt,
                    model,X_testt,
                    warm_start_weights,
                    early_stopping,tol,
                    lrs[k],lr_decay,
                    alphas[j],reg_type,thresh,
                    num_epochs,X_allt.shape[0],
                    num_neighbors,random_val,
                    False,save_state,case_id)
                lrr = linregress((y_testt,np.squeeze(yh.detach().cpu().numpy())))
                err[i,j,k] = lrr.rvalue
        print('Fold',str(i),'complete')
    E = np.mean(err,axis=0)
    js,ks = np.where(E==np.amax(E))
    if len(js) > 1 or len(ks) > 1:
        print('Non-unique regularization parameter')
        js = js[0]
        ks = ks[0]
    print('Retraining model with regularization parameter',str(alphas[js]),'which scored',str(np.mean(err[:,js])))
    y_out, model, _ = train_model(X_all,y_all,
                                model,X_test,warm_start_weights,
                                early_stopping,tol,
                                float(lrs[k]),lr_decay,
                                float(alphas[js]),reg_type,thresh,
                                num_epochs,batch_size,
                                num_neighbors,random_val,
                                2,save_state,case_id)
    return y_out, model, _, alphas[js]
