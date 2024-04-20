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
from multiprocessing import Pool, set_start_method, get_start_method, freeze_support
from numpy import matlib
from scipy import ndimage
from util import full_path
from util import extract
from loader import data_loader
from IPython.display import HTML
import time
# Load data
debug = 1
reload = 1
suffix = '115_'
# Provide paths
segs = full_path('/home/ali/RadDBS-QSM/data/nii/new_seg/')
qsms = full_path('/home/ali/RadDBS-QSM/data/nii/qsm/')
case_id = []
voxel_size = ((0.5, 0.5, 0.5))
# Save location
npy_dir = '/home/ali/RadDBS-QSM/data/npy/new/'
phi_dir = '/home/ali/RadDBS-QSM/data/phi/new/'
roi_path = '/home/ali/RadDBS-QSM/data/xlxs/new_segs.csv'
# Save case IDs
for cases in qsms:
    if os.path.isfile('/home/ali/RadDBS-QSM/data/nii/new_seg/seg_'+cases[-9:-7]+'.nii.gz'):
        print('Found','/home/ali/RadDBS-QSM/data/nii/new_seg/seg_'+cases[-9:-7]+'.nii.gz')
        case_id.append(cases[-9:-7])
        if debug == 1:
            extract(cases,'/home/ali/RadDBS-QSM/data/nii/new_seg/seg_'+cases[-9:-7]+'.nii.gz',
                npy_dir,phi_dir,roi_path,cases[-9:-7],suffix,voxel_size,False,False,False,[0,1,2,5,6,7,8])
    else:
        print('Missing','/home/ali/RadDBS-QSM/data/nii/new_seg/seg_'+cases[-9:-7]+'.nii.gz')

# Parallel extraction
if debug == 0:
    packet = [*zip(qsms[:],segs[:],[npy_dir]*len(case_id),
                [phi_dir]*len(case_id),
                [roi_path]*len(case_id),
                case_id[:],[suffix]*len(case_id),
                [voxel_size]*len(case_id),
                [False]*len(case_id),
                [False]*len(case_id),   
                [False]*len(case_id),
                [[0,1,2,5,6,7,8]]*len(case_id))]

    pool = Pool(processes=8)

    results = pool.starmap(extract,packet)






