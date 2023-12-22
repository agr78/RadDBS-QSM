# %%
# Import libraries
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
from notebook import notebookapp
from numpy import matlib
from scipy import ndimage
from util import pyvis
from util import extract
from loader import data_loader
from IPython.display import HTML
from util import extract

# Load data
reload = 0
suffix = '90'
segs, qsms, n_cases, case_list = data_loader('/media/mts_dbs/dbs/all/nii/qsm/','/media/mts_dbs/dbs/all/nii/seg/',reload,suffix,'QSM_e10_imaginary_')
lines = case_list.read()
lists = np.loadtxt(case_list.name, comments="#", delimiter=",", unpack=False,dtype=str)
case_id = []
for lines in lists:     
    case_id.append(lines[-9:-7])

npy_dir = '/media/mts_dbs/dbs/all/npy/'
phi_dir = '/media/mts_dbs/dbs/all/phi/'
roi_path = '/data/Ali/atlas/mcgill_pd_atlas/PD25-subcortical-labels.csv'

# packet = [*zip(qsms[41:60],segs[41:60],[npy_dir]*len(case_id[41:60]),[phi_dir]*len(case_id[41:60]),[roi_path]*len(case_id[41:60]),case_id[41:60],[suffix]*len(case_id[41:60]))]
# pool = Pool(os.cpu_count()-5)
# results = pool.starmap(extract,packet)

packet = [*zip(qsms[:],segs[:],[npy_dir]*len(case_id[:]),[phi_dir]*len(case_id[:]),[roi_path]*len(case_id[:]),case_id[:],[suffix]*len(case_id[:]),[True]*len(case_id[:]))]
pool = Pool(os.cpu_count()-5)
results = pool.starmap(extract,packet)

# packet = [*zip(qsms[0:20],segs[0:20],[npy_dir]*len(case_id[0:20]),[phi_dir]*len(case_id[0:20]),[roi_path]*len(case_id[0:20]),case_id[0:20],[suffix]*len(case_id[0:20]))]
# pool = Pool(os.cpu_count()-5)
# results = pool.starmap(extract,packet)

# packet = [*zip(qsms[21:40],segs[21:40],[npy_dir]*len(case_id[21:40]),[phi_dir]*len(case_id[21:40]),[roi_path]*len(case_id[21:40]),case_id[21:40],[suffix]*len(case_id[21:40]))]
# pool = Pool(os.cpu_count()-5)
# results = pool.starmap(extract,packet)




