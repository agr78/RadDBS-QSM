# %%
# Import libraries
import sys
sys.path.append('../')
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import nibabel as nib
from sklearn.svm import SVR
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
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
from multiprocessing import Pool, set_start_method
from numpy import matlib
from scipy import ndimage
from util import full_path
from util import extract
from loader import data_loader
from IPython.display import HTML

# Load data
reload = 1
suffix = 'chh_lr_rft'
voxel_size = ((0.9,0.9,0.9))
debug = 0
# Provide paths
segs = full_path('/home/ali/RadDBS-QSM/data/nii/chh/orig/seg')
qsms = full_path('/home/ali/RadDBS-QSM/data/nii/chh/lr_roi_ft/')
npy_dir = '/home/ali/RadDBS-QSM/data/npy/chh/batch/'
phi_dir = '/home/ali/RadDBS-QSM/data/phi/chh/batch/'
roi_path = '/home/ali/RadDBS-QSM/data/xlxs/chh_rois.csv'
case_id = []
# Save case IDs
for cases in segs:
    case_id.append(cases[43:45])
    if debug == 1:
        extract('/home/ali/RadDBS-QSM/data/nii/chh/orig/qsm/'+'QSM_lrr_'+cases[-13:-11]+'.nii',cases,
            npy_dir,phi_dir,roi_path,cases[43:45],suffix,voxel_size,False,False,False,False)

if debug == 0:
    # Parallel extraction
    packet = [*zip(qsms[:],segs[:],[npy_dir]*len(case_id),
                [phi_dir]*len(case_id),
                [roi_path]*len(case_id),
                case_id[:],[suffix]*len(case_id),
                [voxel_size]*len(case_id),
                [False]*len(case_id),
                [False]*len(case_id),   
                [False]*len(case_id),
                [True]*len(case_id))]
    pool = Pool(processes=1)
    results = pool.starmap(extract,packet)



