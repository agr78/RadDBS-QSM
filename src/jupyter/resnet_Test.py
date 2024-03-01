# %%
import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.preprocessing as skp
import sklearn.feature_selection as skf
from scipy.stats import linregress
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import util
from imbalanced_regression.resnet import resnet50

# %%
# Get case IDs
case_list = open('/home/ali/RadDBS-QSM/data/docs/cases_90','r')
lines = case_list.read()
lists = np.loadtxt(case_list.name,comments="#", delimiter=",",unpack=False,dtype=str)
case_id = []
for lines in lists:     
    case_id.append(lines[-9:-7])

# Load scores
file_dir = '/home/ali/RadDBS-QSM/data/docs/QSM anonymus- 6.22.2023-1528.csv'
motor_df = util.filter_scores(file_dir,'pre-dbs updrs','stim','CORNELL ID')
# Find cases with all required scores
subs,pre_imp,post_imp,pre_updrs_off = util.get_full_cases(motor_df,
                                                          'CORNELL ID',
                                                          'OFF (pre-dbs updrs)',
                                                          'ON (pre-dbs updrs)',
                                                          'OFF meds ON stim 6mo')
# Load extracted features
npy_dir = '/home/ali/RadDBS-QSM/data/npy/'
phi_dir = '/home/ali/RadDBS-QSM/data/phi/phi/'
roi_path = '/data/Ali/atlas/mcgill_pd_atlas/PD25-subcortical-labels.csv'
n_rois = 6
Phi_all, X_all, R_all, K_all, ID_all = util.load_featstruct(phi_dir,npy_dir+'X/',npy_dir+'R/',npy_dir+'K/',n_rois,1595,False)
ids = np.asarray(ID_all).astype(int)
# Find overlap between scored subjects and feature extraction cases
c_cases = np.intersect1d(np.asarray(case_id).astype(int),np.asarray(subs).astype(int))
# Complete case indices with respect to feature matrix
c_cases_idx = np.in1d(ids,c_cases)
X_all_c = X_all[c_cases_idx,:,:]
K_all_c = K_all[c_cases_idx,:,:]
R_all_c = R_all[c_cases_idx,:,:]
# Re-index the scored subjects with respect to complete cases
s_cases_idx = np.in1d(subs,ids[c_cases_idx])
subsc = subs[s_cases_idx]
pre_imp = pre_imp[s_cases_idx]
post_imp = post_imp[s_cases_idx]
pre_updrs_off = pre_updrs_off[s_cases_idx]
per_change = post_imp

# %%
X_train,X_test,y_train,y_test,train_index,test_index = util.set_split(X_all_c,per_change,1,6/len(X_all_c))
X0_ss0,scaler_ss,X_test_ss0 = util.model_scale(skp.StandardScaler(),
                                             X_train,train_index,X_test,test_index,pre_updrs_off)

# %%
# Weight initialization
def init_weights(m):
    if type(m) == nn.Linear:
        m.weight.data.fill_(0)

# %%
def net(X_train,y_train,n_epochs,alpha,beta):   
    # Convert to 2D PyTorch tensors
    idx_val = np.random.randint(0,len(y_train))
    print(idx_val)
    X_val = torch.tensor(X_train[idx_val,:], dtype=torch.float32)
    y_val = torch.tensor(y_train[idx_val], dtype=torch.float32).reshape(-1, 1)
    X_train = torch.tensor(np.delete(X_train,idx_val,axis=0), dtype=torch.float32)
    y_train = torch.tensor(np.delete(y_train,idx_val), dtype=torch.float32).reshape(-1, 1)

    model = resnet50(X_val,fds=True, bucket_num=100, bucket_start=3,
                     start_update=0, start_smooth=1,
                     kernel='gaussian', ks=9, sigma=1, momentum=0.9, targets=y_train, epoch=0)

    #model.apply(init_weights)
    # loss function and optimizer
    loss_fn = nn.MSELoss()  # mean square error
    optimizer = optim.SGD(model.parameters(),lr=alpha,momentum=beta)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',factor=1e-1,patience=5,min_lr=1e-12)
    batch_size = 1  # size of each batch
    batch_start = torch.arange(0, len(X_train), batch_size)
    
    # Hold the best model
    best_mse = np.inf   # init to infinity
    best_weights = None
    history = []
    training_loss = []
    batch_loss = []
    for epoch in range(n_epochs):
        # if best_mse > 0.05:
                # Define the model
            model = resnet50(X_val,fds=True, bucket_num=100, bucket_start=3,
                     start_update=0, start_smooth=1,
                     kernel='gaussian', ks=9, sigma=1, momentum=0.9, targets=y_train, epoch=epoch)
            model.train()
            with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
                bar.set_description(f"Epoch {epoch}")
                for start in bar:
                    # take a batch
                    X_batch = X_train[start:start+batch_size,:]
                    y_batch = y_train[start:start+batch_size]
                    # forward pass
                    y_pred,X_batch_s = model.forward(X_batch.T)
                    print(y_pred.shape)
                    print(X_batch_s.shape)
                    loss = loss_fn(y_pred, y_batch)
                    batch_loss.append(loss.detach())
                    # backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    # update weights
                    optimizer.step()
                    # print progress
                    bar.set_postfix(mse=float(loss))
    
            # evaluate accuracy at end of each epoch
            model.eval()
            training_loss.append(np.mean(batch_loss))
            y_pred = model(X_val.T)
            mse = loss_fn(y_pred, y_val)
            mse = float(mse)
            history.append(mse)
            scheduler.step(mse)
            if mse < best_mse:
                best_mse = mse
                print('Best MSE:',best_mse,'at learning rate',optimizer.param_groups[0]['lr'])
                best_weights = copy.deepcopy(model.state_dict())
                print(y_val)
    
    # restore model and return best accuracy
    model.load_state_dict(best_weights)
    print("MSE: %.2f" % best_mse)
    print("RMSE: %.2f" % np.sqrt(best_mse))

    plt.plot(history)
    plt.plot(training_loss)
    
    plt.show()
    return model

# %%
results = np.zeros(per_change.shape)

# %%
for j in np.arange(len(subsc)):
    # Split the data
    Js = []
    test_id = subsc[j]
    test_index = subsc == test_id
    train_index = subsc != test_id
    X_train = X_all_c[train_index,:,:]
    X_test = X_all_c[test_index,:,:]
    y_train = per_change[train_index]
    y_test = per_change[test_index]

    # Cross validation
    X0_ss0,scaler_ss,X_test_ss0 = util.model_scale(skp.StandardScaler(),
                                                X_train,train_index,X_test,test_index,pre_updrs_off)
    # # Feature selection
    # with np.errstate(divide='ignore', invalid='ignore'):
    #     sel = skf.SelectKBest(skf.r_regression,k=2000)
    #     X0_ss = sel.fit_transform(X0_ss0,y_train)
    #     X_test_ss = sel.transform(X_test_ss0.reshape([X_test_ss0.shape[0],
    #                                             X_test_ss0.shape[1]*X_test_ss0.shape[2]]))
    tmodel = net(X0_ss0,y_train,20,0.1,0.9)
    
    results[j] = tmodel(torch.tensor(X_test_ss0.reshape(X_test_ss0.shape[0],X_test_ss0.shape[1]*X_test_ss0.shape[2]),dtype=torch.float32))
    # Output results
    print('Model predicts',str(results[j]),'for case',str(test_id),'with actual improvement',str(per_change[j]))



# %%
[fig,ax] = plt.subplots()
lr_prepost = linregress(results,per_change)
ax.scatter(results,per_change)
ax.plot(results,results*lr_prepost.slope+lr_prepost.intercept,'-r')
ax.set_title('Model performance')
ax.set_ylabel("DBS improvement")
ax.set_xlabel("Prediction")
text = f"$y={lr_prepost.slope:0.2f}\; x{lr_prepost.intercept:+0.2f}$\n$r = {lr_prepost.rvalue:0.2f}$\n$p = {lr_prepost.pvalue:0.3f}$"
ax.text(0.35, 0.75, text,transform=ax.transAxes,
    fontsize=14, verticalalignment='top')
ax.hlines(0.3,0,2,linestyle='dashed',color='black')
ax.vlines(0.3,0,2,linestyle='dashed',color='black')

# %%
results

# %%
y_test


