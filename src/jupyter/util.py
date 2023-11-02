# Import libraries
import matplotlib.pyplot as plt
import matplotlib as mpl
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
import pickle

def pyvis(figx,figy):
    plt.style.use('dark_background')
    plt.rcParams["figure.figsize"] = (figx,figy)
    def remove_keymap_conflicts(new_keys_set):
        for prop in plt.rcParams:
            if prop.startswith('keymap.'):
                keys = plt.rcParams[prop]
                remove_list = set(keys) & new_keys_set
                for key in remove_list:
                    keys.remove(key)

    def multi_slice_viewer(volume):
        remove_keymap_conflicts({'j', 'k'})
        fig, ax = plt.subplots()
        ax.volume = volume
        ax.index = volume.shape[0]//2
        ax.imshow(volume[ax.index])
        fig.canvas.mpl_connect('key_press_event', process_key)

    def process_key(event):
        fig = event.canvas.figure
        ax = fig.axes[0]
        if event.key == 'j':
            previous_slice(ax)
        elif event.key == 'k':
            next_slice(ax)
        fig.canvas.draw()
        
    def previous_slice(ax):
        volume = ax.volume
        ax.index = (ax.index-1) % volume.shape[0] 
        ax.images[0].set_array(volume[ax.index])

    def next_slice(ax):
        volume = ax.volume
        ax.index = (ax.index+1) % volume.shape[0]
        ax.images[0].set_array(volume[ax.index])

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

def imfeat_plotter(lower_idx,upper_idx,title_string):
    plt.close('all') 
    fig, ax = plt.subplots(figsize = (15,10))
    if title_string == 'wavelet':
        L = (upper_idx-lower_idx)
        print(L)
        cax = ax.imshow(((Mc_lrs[lower_idx:upper_idx]*Mi_lrs[lower_idx:upper_idx,:]).T).reshape((6,L/6)), interpolation='nearest', 
            cmap='Spectral', vmin=0, vmax=1)
    else:
        cax = ax.imshow((Mc_lrs[lower_idx:upper_idx]*Mi_lrs[lower_idx:upper_idx,:]).T, interpolation='nearest', 
                    cmap='Spectral', vmin=0, vmax=1)
    ax.set_xlabel('Feature index')
    ax.set_ylabel('ROI index')
    plt.xticks(np.arange(upper_idx-lower_idx))
    plt.yticks(np.arange(Mc_vds.shape[1]))
    ax.set_xticklabels(Kcs[lower_idx:upper_idx,0], rotation='vertical', fontsize=7)
    ax.set_yticklabels(R_rs[0][0],rotation='horizontal',fontsize=7)
    plt.title('Robustness $\\rho$ between ground truth and LARO for '+title_string+' image')
    divider = make_axes_locatable(ax)
    caxf = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(cax, cax=caxf, orientation='vertical',ticks=[0, 0.5, 1])
    plt.show()

def extract(qsm,seg,sub_in,pre_dbs_off_meds_in,npy_dir,phi_dir,suffix):
    fv_count = 0
    seg_labels_all = [0,1,2,3,4,5,6,7]
    Phi_gt = []
    x_row_gt = []
    keylib = []
    roilib = []
    roi_names = []
    voxel_size = ((0.5,0.5,0.5))
    seg_sitk = sitk.GetImageFromArray(seg)
    seg_sitk.SetSpacing(voxel_size)
    qsm_sitk_gt = sitk.GetImageFromArray(qsm)
    qsm_sitk_gt.SetSpacing(voxel_size)
    for j in seg_labels_all:
        if 0 < j < 7:
            fv_count = 0
            featureVector_gt = extractor.execute(qsm_sitk_gt,seg_sitk,label=int(j));
            Phi_gt.append(featureVector_gt)
            for key, value in six.iteritems(featureVector_gt):
                if 'diagnostic' in key:
                    next
                else:
                    x_row_gt.append(featureVector_gt[key])
                    fv_count = fv_count+1
                    keylib.append(key)
                    roilib.append(j)
                    mask = np.row_stack([roi_df[row].str.contains(str(int(roilib[-1])), na = False) for row in roi_df])
                    roi_names.append(np.asarray(roi_df.iloc[mask.any(axis=0),1])[0])
            x_row_gt.append(pre_dbs_off_meds_in)
            fv_count = fv_count+1
            print('Extracting features for subject',sub_in,
                'ROI',j,'and appending feature matrix with vector of length',
                fv_count,'with UPDRS score',pre_dbs_off_meds_in)

    X0_gt = np.array(x_row_gt)
    npy_file = npy_dir+'X_'+suffix+str(sub_in)+'.npy'
    np.save(npy_file,X0_gt)
    K = np.asarray(keylib)
    R = np.asarray(roi_names)
    K_file = npy_dir+'K_'+suffix+str(sub_in)+'.npy'
    R_file = npy_dir+'R_'+suffix+str(sub_in)+'.npy'
    np.save(K_file,K)
    np.save(R_file,R)
    Phi_file = phi_dir+'Phi_'+suffix+str(sub_in)
    print('Saving ground truth feature vector')
    with open(Phi_file, 'wb') as fp:  
        pickle.dump(Phi_gt, fp)
    
def window3D(w):
    # Convert a 1D filter kernel to 3D
    L=w.shape[0]
    m1=np.outer(np.ravel(w), np.ravel(w))
    win1=np.tile(m1,np.hstack([L,1,1]))
    m2=np.outer(np.ravel(w),np.ones([1,L]))
    win2=np.tile(m2,np.hstack([L,1,1]))
    win2=np.transpose(win2,np.hstack([1,2,0]))
    win=np.multiply(win1,win2)
    return win

def k_crop(im,factor,win):
    (cx,cy,cz) = (im.shape[0]//factor,im.shape[1]//factor,im.shape[2]//factor)
    win = win[(cx-(cx//2)):(cx+(cx//2)),(cy-(cy//2)):(cy+(cy//2)),(cz-(cz//2)):(cz+(cz//2))]
    print('Cropping to',str(cx),str(cy),str(cz),'from',str((cx-(cx//2))),'to',str((cx+(cx//2))),'and',str((cy-(cy//2))),'to',str((cy+(cy//2))),'and',str((cz-(cz//2))),'to',str((cz+(cz//2))))
    im_k = np.real(np.fft.ifftn(np.fft.ifftshift(win*(np.fft.fftshift(np.fft.fftn(im))[(cx-(cx//2)):(cx+(cx//2)),(cy-(cy//2)):(cy+(cy//2)),(cz-(cz//2)):(cz+(cz//2))]))))
    plt.imshow(im_k[:,:,(cz//2)-10])
    return im_k

def load_featstruct(phi_directory,X_directory,R_directory,K_directory,n_cases,n_rois,n_features):
    # Initialize output arrays
    X_all = np.zeros((n_cases,n_rois,n_features))
    R_all = np.zeros((n_cases,n_rois,n_features-1)).astype(str)
    K_all = np.zeros((n_cases,n_rois,n_features-1)).astype(str)

    # Convert directories to lists
    phi_directory_struct = os.listdir(phi_directory)
    X_directory_struct  = os.listdir(X_directory)
    R_directory_struct  = os.listdir(R_directory)
    K_directory_struct  = os.listdir(K_directory)

    for feature_matrix in phi_directory_struct:
        with open(phi_directory+feature_matrix, "rb") as fp:  
            Phi_case = pickle.load(fp)
            Phi_all.append(Phi_case)

    # Load feature arrays
    count = 0
    for feature_array in X_directory_struct:
        X_case = np.load(X_directory+feature_array)
        X_all[count,:,:] = X_case.reshape((n_rois,n_features)).transpose((0,1))
        count = count+1

    # Load ROI indices
    count = 0
    for feature_roi in R_directory_struct:
        R_case = np.load(R_directory+feature_roi)
        R_all[count,:,:] = R_case.reshape((n_rois,n_features-1)).transpose((0,1))
        count = count+1

    # Load key indices
    count = 0
    for feature_key in K_directory_struct:
        K_case = np.load(K_directory+feature_key)
        K_all[count,:,:] = K_case.reshape((n_rois,n_features-1)).transpose((0,1))
        count = count+1

    return Phi_all, X_all, R_all, K_all

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def l_curve(base_min,base_max,X,y,n_points):
    fig,ax = plt.subplots()
    alphas = np.logspace(base_min,base_max,n_points)
    solution_norm = []
    residual_norm = []
    idx = np.where(alphas==find_nearest(alphas,1e-4))
    for alpha in alphas: 
        lm = Lasso(alpha=alpha,max_iter=10000)
        lm.fit(X, y)
        solution_norm += [(lm.coef_**2).sum()]
        residual_norm += [((lm.predict(X) - y)**2).sum()]

    plt.loglog(residual_norm, solution_norm, 'k-')
    plt.show()