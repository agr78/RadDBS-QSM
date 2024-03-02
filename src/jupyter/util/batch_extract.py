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
reload = 1
suffix = '90'
segs, qsms, n_cases, case_list = data_loader('/home/ali/RadDBS-QSM/data/nii/qsm/','/home/ali/RadDBS-QSM/data/nii/seg/',reload,suffix,'QSM_e10_imaginary_')
lines = case_list.read()
lists = np.loadtxt(case_list.name, comments="#", delimiter=",", unpack=False,dtype=str)
case_id = []
for lines in lists:     
    case_id.append(lines[-9:-7])

npy_dir = '/home/ali/RadDBS-QSM/data/npy/'
phi_dir = '/home/ali/RadDBS-QSM/data/phi/'
roi_path = '/home/ali/RadDBS-QSM/data/atlas/mcgill_pd_atlas/PD25-subcortical-labels.csv'

packet = [*zip(qsms[:],segs[:],[npy_dir]*len(case_id[:]),
               [phi_dir]*len(case_id[:]),[roi_path]*len(case_id[:]),case_id[:],[suffix]*len(case_id[:])),False,True]
pool = Pool(8)
results = pool.starmap(extract,packet)





