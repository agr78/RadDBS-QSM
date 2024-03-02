# Import libraries
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
import multiprocessing
from multiprocess import Pool
import time
from joblib import Parallel, delayed
import pandas as pd
import os



# Dark mode
plt.style.use('dark_background')

# PyVis()
class IndexTracker(object):
    def __init__(self, ax, X, sp=None):
        self.ax = ax
        self.X = X
        self.sp = 1 if sp is not None else 0
        rows, cols, self.slices = X.shape
        self.ind = self.slices//2
        self.im = ax.imshow(self.X[:, :, self.ind],cmap=plt.colormaps['gray'],clim=(np.min(X),np.max(X)))

        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(self.im, cax=cax, orientation='vertical')
        self.update()

    def onscroll(self, event):
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.X[:, :, self.ind])
        if self.sp == 0:
            self.ax.set_ylabel('Slice Number: %s' % self.ind)
        else:
            self.ax.set_ylabel('')
        self.im.axes.figure.canvas.draw()
        self.ax.set_xticks([])
        self.ax.set_yticks([])

#fig, ax = plt.subplots(1,1)


# Parallelize feature extraction loop  
def feat_extract(extractor,n_cases,qsm,seg,voxel_size,i):
    Phi = []
    seg_sitk = sitk.GetImageFromArray(seg)
    seg_sitk.SetSpacing(voxel_size.tolist())
    qsm_sitk = sitk.GetImageFromArray(qsm)
    qsm_sitk.SetSpacing(voxel_size.tolist())
    for j in range(1,int(np.max(seg))+1):
        featureVector = extractor.execute(qsm_sitk,seg_sitk,label=j)
        Phi.append(featureVector)
    print('Features extracted for case ' + str(i) + ' of ' + str(n_cases))
    return Phi
 


def par_feat_extract(extractor,n_cases,qsms,segs,voxel_sizes):
    Phi = []
    Phi_i = Parallel(n_jobs=-1,verbose=100)(delayed(feat_extract(extractor,n_cases,qsms[i],segs[i],voxel_sizes[i],i)) for i in range(n_cases))
    Phi.append(Phi_i)


def exclude_outliers(x):
    q3 = np.percentile(x,75)
    q1 = np.percentile(x,25)
    iqr  = q3-q1
    dq = 1.5*iqr
    x_m = np.logical_and(x<=(q3+dq),x>=(q1-dq))
    print('Upper bound of',str(q3+dq))
    print('Lower bound of',str(q1-dq))
    x_out = x[x_m>0]
    print('Excluded',str(max(x.shape)-max(x_out.shape)),'outliers')
    return x_m

def iqr_exclude(x):
    q3 = np.percentile(x,75)
    q1 = np.percentile(x,25)
    print('75th percentile at',str(q3))
    print('25th percentile at',str(q1))
    iqr  = q3-q1
    dq = iqr
    x_m = np.logical_and(x<=(q3),x>=(q1))
    print('Interquartile range of',str(iqr))
    print('Upper bound of',str(q3+dq))
    print('Lower bound of',str(q1-dq))
    x_out = x[x_m>0]
    print('Excluded',str(max(x.shape)-max(x_out.shape)),'outliers')
    return x_m


def scores_df(file_dir,csv_name,header,ColumnName1,ColumnName2):

    # Load patient data
    os.chdir(file_dir)
    df = pd.read_csv(csv_name)

    # Make a copy
    dfd = df.copy()

    # Drop blank columns
    for (columnName, columnData) in dfd.iteritems():
        if columnData.isnull().all():
            print('Dropping NaN column at',columnName)
            dfd.drop(columnName,axis=1,inplace=True)

    # Add relevant column names from headers
    df_headers = []
    for (columnName, columnData) in dfd.iteritems():
        if 'Unnamed' not in columnName:
            df_headers.append(columnName)
        else:
            print('Renaming',columnName,'as',df_headers[-1]+' '+str(dfd.iloc[0, df.columns.get_loc(columnName)-1]))
            dfd.rename(columns={columnName:df_headers[-1]+' '+str(dfd.iloc[0, df.columns.get_loc(columnName)-1])},inplace=True)

    # Make a copy for motor symptoms
    df_out = dfd.copy()
    # Drop non-motor (III) columns
    for (columnName, columnData) in dfd.iteritems():
        if header in columnName:
            next
        elif 'Anonymous ID' in columnName:
            df_out.iloc[0,0] = 'Anonymous ID'
        else:
            df_out.drop(columnName,axis=1,inplace=True)

    # Rename columns with specific metrics
    df_out.columns = df_out.iloc[0]
    df_out = df_out.tail(-1)

    # Convert columns to numerical arrays
    score1 = df_out[ColumnName1].to_numpy().astype('float')
    score2 = df_out[ColumnName2].to_numpy().astype('float')

    # Find numerical entries only
    cases = []
    anon_ids = df_out['Anonymous ID'].to_numpy().astype('float')
    for ids in np.arange(0,score1.__len__()):
        if ~np.isnan(score1[ids]) and ~np.isnan(score2[ids]): 
            cases.append(anon_ids[ids])

    return df_out,anon_ids,cases
