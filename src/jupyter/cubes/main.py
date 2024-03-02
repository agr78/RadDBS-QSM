# %%
# Import libraries
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import nibabel as nib
from sklearn.svm import SVR
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import r2_score
import SimpleITK as sitk
import six
from radiomics import featureextractor 
import numpy as np
import os
import pickle
import pandas as pd
import logging
from scipy.stats import linregress
import smogn
import pandas
from collections import Counter
from multiprocessing import Pool
from notebook import notebookapp
from numpy import matlib
import random
from scipy import ndimage
from util import pyvis
from util import extract
from loader import data_loader
from IPython.display import HTML
import util

# %%
HTML('''
<style>
.jupyter-matplotlib {
    background-color: #000;
}

.widget-label, .jupyter-matplotlib-header{
    color: #fff;
}

.jupyter-button {
    background-color: #333;
    color: #fff;
}
</style>
''')

# %%
# Load data
reload = 0
suffix = '90'
segs, qsms, n_cases, case_list = data_loader('/media/mts_dbs/dbs/all/nii/qsm/',
                                             '/media/mts_dbs/dbs/all/nii/seg/',
                                             reload,suffix,'QSM_e10_imaginary_')


x = random.sample(range(1,n_cases),5)
for j in x:
    QSMs = np.asarray(qsms[j]+20*segs[j])
    pyvis(np.flip(QSMs.T),5,5)

# %%
# Get case IDs
lines = case_list.read()
lists = np.loadtxt(case_list.name,comments="#", delimiter=",",unpack=False,dtype=str)
case_id = []
for lines in lists:     
    case_id.append(lines[-9:-7])

# %%
# Load extracted features
npy_dir = '/media/mts_dbs/dbs/all/npy/'
phi_dir = '/media/mts_dbs/dbs/all/phi/'
roi_path = '/data/Ali/atlas/mcgill_pd_atlas/PD25-subcortical-labels.csv'
n_rois = 6
Phi_all, X_all, R_all, K_all, ID_all = util.load_featstruct(phi_dir,npy_dir+'X/',npy_dir+'R/',npy_dir+'K/',n_rois,1595)
ids = np.asarray(ID_all).astype(int)

# %%
# Load scores
file_dir = '/data/Ali/RadDBS-QSM/src/csv/QSM anonymus- 6.22.2023-1528.csv'
motor_df = util.filter_scores(file_dir,'pre-dbs updrs','stim','CORNELL ID')
# Find cases with all required scores
subs,pre_imp,post_imp,pre_updrs_off = util.get_full_cases(motor_df,
                                                          'CORNELL ID',
                                                          'OFF (pre-dbs updrs)',
                                                          'ON (pre-dbs updrs)',
                                                          'OFF meds ON stim 6mo')
# Find overlap between scored subjects and feature extraction cases
c_cases = np.intersect1d(np.asarray(case_id).astype(int),np.asarray(subs).astype(int))
# Complete case indices with respect to feature matrix
c_cases_idx = np.in1d(ids,c_cases)
X_all_c = X_all[c_cases_idx,:,:]
# Re-index the scored subjects with respect to complete cases
s_cases_idx = np.in1d(subs,ids[c_cases_idx])
pre_imp = pre_imp[s_cases_idx]
post_imp = post_imp[s_cases_idx]
pre_updrs_off = pre_updrs_off[s_cases_idx]
per_change = post_imp

# %%
# Split training and testing data
Xos,oscaler = util.make_feature_matrix(X_all_c,pre_updrs_off)
sss = ShuffleSplit(n_splits=1, test_size=0.2)
sss.get_n_splits(X_all_c,per_change)
train_index, test_index = next(sss.split(X_all_c,per_change))
X_train,X_test = X_all_c[train_index], X_all_c[test_index] 
per_change_train,per_change_test = per_change[train_index], per_change[test_index]

# %%
# Append metric of interest and normalize
Xrs,scaler = util.make_feature_matrix(X_train,pre_updrs_off[train_index])
Xts = util.scale_feature_matrix(X_test,pre_updrs_off[test_index],scaler)

# %%
# Apply SMOGN
X_smogn,y_smogn,idx_kept,sscaler = util.rad_smogn(Xrs,
                                                  per_change_train,
                                                  np.min(per_change_train),
                                                  np.mean(per_change_train),
                                                  1,0,0.5)


# %%
reg = LassoCV(cv=X_smogn.shape[0]-1,n_jobs=-1,verbose=True,max_iter=1e5,eps=1e-9).fit(X_smogn,y_smogn)
reg.score(X_smogn,y_smogn)

# %%
reg.predict(Xts)

# %%
per_change_test

# %%
# util.l_curve(-9,6,X_smogn,post_imp[:len(X_smogn)],100)


