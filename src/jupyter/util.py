# Import libraries
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from numpy import matlib as mb
import SimpleITK as sitk
from tqdm import tqdm
from radiomics import featureextractor 
from joblib import Parallel, delayed
import pandas as pd
import os
import pickle
import six
import logging
import pandas as pd
from sklearn.linear_model import Lasso
import sklearn.model_selection as sms
import sklearn.preprocessing as skp
from sklearn.neighbors import NearestNeighbors
import scipy
from scipy.stats import linregress
import scipy.stats as stats
import smogn
from smogn.phi import phi
from smogn.phi_ctrl_pts import phi_ctrl_pts
import warnings
import sys
import collections
import math
import nibabel as nib
import torch
from torch import nn
from PIL import Image
import re
import nrrd
try:
    from torchvision.models.feature_extraction import create_feature_extractor
except:
    print('Skipping torch import')

def blockPrint():
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    sys.stdout = sys.__stdout__

def remove_keymap_conflicts(new_keys_set):
    for prop in plt.rcParams:
        if prop.startswith('keymap.'):
            keys = plt.rcParams[prop]
            remove_list = set(keys) & new_keys_set
            for key in remove_list:
                keys.remove(key)

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
    ax.set_title('Slice '+str(ax.index))


def next_slice(ax):
    volume = ax.volume
    ax.index = (ax.index+1) % volume.shape[0]
    ax.images[0].set_array(volume[ax.index])
    ax.set_title('Slice '+str(ax.index))

def pyvis(volume,figx,figy,colormap,w,l,cmin,cmax):
    plt.style.use('dark_background')
    plt.rcParams["figure.figsize"] = (figx,figy)
    remove_keymap_conflicts({'j', 'k'})
    fig, ax = plt.subplots()
    ax.volume = volume
    ax.index = volume.shape[0]//2
    if w == 0 and l == 0:
        im = ax.imshow(volume[ax.index],cmap=colormap)
    else:
        volume[volume<-((w-l)//2)] = -(w-l)//2
        volume[volume>((w-l)//2)] = (w-l)//2
        im = ax.imshow(volume[ax.index],cmap=colormap,vmin=cmin, vmax=cmax)
   
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    ax.set_xticks([])
    ax.set_yticks([])
    fig.canvas.mpl_connect('key_press_event', process_key)
    ax.set_axis_off()
    plt.show()
    
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

def imfeat_plotter(Mc_lrs,Mi_lrs,Mc_vds,Kcs,R_rs,lower_idx,upper_idx,title_string):
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

def extract(qsm,seg,npy_dir,phi_dir,roi_path,sub_in,suffix,voxel_size,slices,mask_erode,strict_shape,seg_labels_all):
    print('Initializing',sub_in)
    # if os.path.isfile(npy_dir+'/X_'+suffix+str(sub_in)+'.npy') == False:
    #     print('Beginning extraction for subject',sub_in)
    # else:
    #     exit('Found feature matrix at: '+npy_dir+'X_'+suffix+str(sub_in)+'.npy')
    if ~isinstance(qsm, np.ndarray):
        qsm = nib.load(qsm).get_fdata()
        seg = nib.load(seg).get_fdata()
    logger = logging.getLogger("radiomics")
    logger.setLevel(logging.ERROR)
    fv_count = 0
    Phi_gt = []
    x_row_gt = []
    keylib = []
    roilib = []
    roi_names = []
    if mask_erode == True:
        print('Eroding mask')
        seg_labels = seg
        seg = scipy.ndimage.morphology.binary_erosion(seg_labels)*seg_labels
    else:
        print('Not eroding mask')
    if slices == True:
        print('Slice-wise extraction')
        npy_dir = npy_dir+'/slices/'
        phi_dir = phi_dir+'/slices/'
        for i in np.arange(seg.shape[2]):
            if np.sum(seg[:,:,i])>1:
                seg_sitk = sitk.GetImageFromArray(seg[:,:,i])
                seg_sitk.SetSpacing(voxel_size)
                qsm_sitk_gt = sitk.GetImageFromArray(qsm[:,:,i])
                qsm_sitk_gt.SetSpacing(voxel_size)
                print('Evaluating slice',str(i),'on case',str(sub_in))
                # Generate feature structure Phi from all ROIs and all cases
                extractor = featureextractor.RadiomicsFeatureExtractor()
                extractor.enableAllFeatures()
                extractor.enableAllImageTypes()
                roi_txt = pd.read_csv(roi_path,header=None)
                roi_df = roi_txt.astype(str)
                seg_labels_all = np.unique(seg[:,:,i])
                for j in seg_labels_all:
                    if 0.0 < j <= np.amax(seg_labels_all):
                        seg_slice = np.asarray(seg[:,:,i])
                        if (np.sum(seg_slice==j) > 1) and (np.max(np.sum((seg_slice==j),axis=0)) > 1) and (np.max(np.sum((seg_slice==j),axis=1)) > 1):
                            fv_count = 0
                            featureVector_gt = extractor.execute(qsm_sitk_gt,seg_sitk,label=int(j))
                            Phi_gt.append(featureVector_gt)
                            for key, value in six.iteritems(featureVector_gt):
                                if 'diagnostic' in key:
                                    next
                                else:
                                    x_row_gt.append(featureVector_gt[key])
                                    fv_count = fv_count+1
                                    keylib.append(key)
                                    roilib.append(j)
                                    mask = np.row_stack([roi_df[int(row)].str.contains(str(int(roilib[-1])), na = False) for row in roi_df])
                                    roi_names.append(np.asarray(roi_df.iloc[mask.any(axis=0),1])[0])
                            print('Extracting features for subject',sub_in,
                                'ROI',j,'and appending feature matrix with vector of length',
                                fv_count)
                # Convert each row to numpy array
                X0_gt = np.array(x_row_gt)
                if mask_erode == True:
                    npy_file = npy_dir+'X_e_'+suffix+str(sub_in)+'_'+str(i)+'.npy'
                    Phi_file = phi_dir+'Phi_e'+str(sub_in)+'_'+str(i)
                    K_file = npy_dir+'K_e'+str(sub_in)+'_'+str(i)+'.npy'
                    R_file = npy_dir+'R_e'+str(sub_in)+'_'+str(i)+'.npy'
                else:
                    npy_file = npy_dir+'X_'+suffix+str(sub_in)+'_'+str(i)+'.npy'
                    Phi_file = phi_dir+'Phi_'+str(sub_in)+'_'+str(i)
                    K_file = npy_dir+'K_'+str(sub_in)+'_'+str(i)+'.npy'
                    R_file = npy_dir+'R_'+str(sub_in)+'_'+str(i)+'.npy'
               
                K = np.asarray(keylib)
                R = np.asarray(roi_names)
                np.save(npy_file,X0_gt)
                np.save(K_file,K)
                np.save(R_file,R)
                
                with open(Phi_file, 'wb') as fp:  
                    pickle.dump(Phi_gt, fp)
    else:
        print('ROI-wise extraction')
        seg_sitk = sitk.GetImageFromArray(seg)
        seg_sitk.SetSpacing(voxel_size)
        qsm_sitk_gt = sitk.GetImageFromArray(qsm)
        qsm_sitk_gt.SetSpacing(voxel_size)
        if strict_shape == True:
            print('QSM shape:',qsm_sitk_gt.GetSize())
            print('Mask shape:',seg_sitk.GetSize())
            if seg_sitk.GetSize() != qsm_sitk_gt.GetSize():
                print('Size mismatch at',sub_in)
                exit()
        else:
            print('Cropping QSM if necessary')
            if seg_sitk.GetSize() != qsm_sitk_gt.GetSize():
                ds = np.amin(((qsm.shape)[-1],(seg.shape)[-1]))
                seg_sitk = sitk.GetImageFromArray(seg[:,:,:ds])
                print(seg_sitk.GetSize())
                qsm_sitk_gt = sitk.GetImageFromArray(qsm[:,:,:ds])
                print(qsm_sitk_gt.GetSize())
        # Generate feature structure Phi from all ROIs and all cases
        extractor = featureextractor.RadiomicsFeatureExtractor()
        extractor.enableAllFeatures()
        extractor.enableAllImageTypes()
        extractor.enableFeatureClassByName('shape2D',enabled=False)
        roi_txt = pd.read_csv(roi_path,header=None)
        roi_df = roi_txt.astype(str)
        print('Beginning label loop')
        for j in seg_labels_all:
            if 0.0 < j <= np.amax(seg_labels_all):
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
                        mask = np.row_stack([roi_df[int(row)].str.contains(str(int(roilib[-1])), na = False) for row in roi_df])
                        roi_names.append(np.asarray(roi_df.iloc[mask.any(axis=0),1])[0])
                print('Extracting features for subject',sub_in,
                    'ROI',j,'and appending feature matrix with vector of length',
                    fv_count)
            
        # Convert each row to numpy array
        X0_gt = np.array(x_row_gt)
        K = np.asarray(keylib)
        R = np.asarray(roi_names)
        if mask_erode == True:
            npy_file = npy_dir+'X_e_'+suffix+str(sub_in)+'.npy'
            Phi_file = phi_dir+'Phi_e'+str(sub_in)
            K_file = npy_dir+'K_e'+str(sub_in)+'.npy'
            R_file = npy_dir+'R_e'+str(sub_in)+'.npy'
        else:
            npy_file = npy_dir+'X_'+suffix+str(sub_in)+'.npy'
            Phi_file = phi_dir+'Phi_'+str(sub_in)
            K_file = npy_dir+'K_'+str(sub_in)+'.npy'
            R_file = npy_dir+'R_'+str(sub_in)+'.npy'
        np.save(npy_file,X0_gt)
        np.save(K_file,K)
        np.save(R_file,R)
        print('Saving feature vectors of size',str(X0_gt.shape))
        with open(Phi_file, 'wb') as fp:  
            pickle.dump(Phi_gt, fp)
    
def window3D(w):
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

def load_featstruct(phi_directory,X_directory,R_directory,K_directory,n_rois,n_features,slices):
    # Initialize output arrays
    Phi_all = []
    ID_all = []
    slice_id = []
    # Convert directories to lists
    phi_directory_struct = sorted(os.listdir(phi_directory))
    X_directory_struct  = sorted(os.listdir(X_directory))
    R_directory_struct  = sorted(os.listdir(R_directory))
    K_directory_struct  = sorted(os.listdir(K_directory))
    print(X_directory_struct)
    # print(K_directory_struct)
    # print(R_directory_struct)
    # print(phi_directory_struct)
    # Load feature dictionary
    for feature_matrix in phi_directory_struct:
        with open(phi_directory+feature_matrix, "rb") as fp:  
            if slices == True:
                print('Check ID all, incorrect!')
                if len(feature_matrix) == 10:
                    ID_all.append(feature_matrix[-6:-4])
                else:
                    ID_all.append(feature_matrix[-5:-3])
                slice_id.append(feature_matrix[-6:])
                if np.mod(len(ID_all),1000) == 0:
                    print('Appended',str(len(ID_all)),'slices')
                Phi_all = []
            else:
                Phi_case = pickle.load(fp)
                Phi_all.append(Phi_case)
                case_ids = str(feature_matrix).split("_",1)[1]
                ID_all.append(case_ids)
    n_cases = len(np.asarray(ID_all))
    X_all = np.zeros((n_cases,n_rois,n_features))
    # print(n_cases)
    # print(n_rois)
    # print(n_features)
    R_all = np.zeros((n_cases,n_rois,n_features)).astype(str)
    K_all = np.zeros((n_cases,n_rois,n_features)).astype(str)
    # Load feature arrays
    count = 0
    print('Allocated arrays')
    if slices == True:
        X_out = []
        for feature_array in X_directory_struct:
            #try:
            X_case = np.load(X_directory+feature_array,allow_pickle=True)
            feature_key = 'K_'+feature_array[4:]
            K_case = np.load(K_directory+feature_key)
            feature_roi = 'R_'+feature_array[4:]
            R_case = np.load(R_directory+feature_roi)
            slice_roi = len(np.unique(R_case))
            if slice_roi == 6:
                X_out.append(X_case.reshape((slice_roi,n_features)).transpose((0,1)))
                R_all = []
                K_all = []
        X_all = X_out
            #except:
            #    print('Failed to load',X_directory+feature_array)
    if slices == False:
        # Load features
        count = 0
        for feature_matrix in X_directory_struct:
            X_case = np.load(X_directory+feature_matrix)
            X_all[count,:,:] = X_case.reshape((n_rois,n_features)).transpose((0,1))
            #print('Loading index',count,'with features from',feature_matrix)
            count = count+1
        # Load ROI indices
        count = 0
        print('Created feature matrix')
        for feature_roi in R_directory_struct:
            R_case = np.load(R_directory+feature_roi)
            R_all[count,:,:] = R_case.reshape((n_rois,n_features)).transpose((0,1))
            count = count+1
        # Load key indices
        count = 0
        print('Created ROI matrix')
        for feature_key in K_directory_struct:
            K_case = np.load(K_directory+feature_key)
            K_all[count,:,:] = K_case.reshape((n_rois,n_features)).transpose((0,1))
            count = count+1
        print('Created feature label matrix')
    else:
        next
    return Phi_all, X_all, R_all, K_all, ID_all

def slice_pick(subsc,per_change,pre_metric,pre_comp,pshape,roi_l,roi_u,mask_crop_output,mask_output,o_index,file_path,qsm_path,seg_prefix,L,new_prefix,save_image,img_directory,visualize,reload):
    qsms = full_path(qsm_path)
    writer = sitk.ImageFileWriter()
    qsms_subs = []
    X_img = []
    seg_prefix_dd = seg_prefix[:-1]
    if reload == True:
        for Q in np.arange(len(qsms)):
            qsms_subs.append(int(qsms[Q][-9:-7]))

        for j in np.arange(len(subsc)):
            q = np.where(qsms_subs==subsc[j])[0][0]
            data = nib.load(qsms[q])
            if qsms_subs[q] < L:
                mask = nib.load(seg_prefix+'0'+str(qsms_subs[q])+'.nii.gz').get_fdata()
            elif new_prefix == True:
                mask = nib.load(seg_prefix_dd+str(qsms_subs[q])+'.nii.gz').get_fdata()
            else:
                mask = nib.load(seg_prefix+str(qsms_subs[q])+'.nii.gz').get_fdata()
            mask[mask > roi_u] = 0
            mask[mask < roi_l] = 0
            maskc = pad_to(mask_crop(mask,mask,0),pshape[0],pshape[1],pshape[2])
            mask_k = []
            k_all = []
            if mask_crop_output == True:
                if mask_output == True:
                    data = pad_to(mask_crop(mask*data.get_fdata(),mask,0),pshape[0],pshape[1],pshape[2])
                else:
                    data = pad_to(mask_crop(data.get_fdata(),mask,0),pshape[0],pshape[1],pshape[2])
                for k in np.arange(maskc.shape[2]):
                    if np.sum(maskc[:,:,k]) > 0:
                        mask_k.append(np.sum(maskc[:,:,k]))
                        k_all.append(k)
            else:     
                if o_index == True:
                    data = data.get_fdata()
                    for k in np.arange(mask.shape[2]):
                        if np.sum(mask[:,:,k]) > 0:
                            mask_k.append(np.sum(mask[:,:,k]))
                            k_all.append(k)
                else:
                    if mask_output == True:
                        data = mask_crop(mask*data.get_fdata(),mask,pshape[0])
                    else:
                        data = mask_crop(data.get_fdata(),mask,pshape[0])
                    maskc = mask_crop(mask,mask,pshape[0])
                    for k in np.arange(maskc.shape[2]):
                        if np.sum(maskc[:,:,k]) > 0 and len(np.unique(maskc[:,:,k]))==5:
                            mask_k.append(np.sum(maskc[:,:,k]))
                            k_all.append(k)
                    
            img = data[:,:,k_all[np.argmax(mask_k)]]
            if save_image == True:
                img_sitk =  sitk.Cast(sitk.GetImageFromArray(img),sitk.UInt32)
                writer.SetFileName(str((img_directory)+'X_img_'+str(j)+'.png'))
                writer.Execute(img_sitk)
            X_img.append(torch.Tensor(img).cuda())

            print('Maximum volume found at slice',str(k_all[np.argmax(mask_k)]),'for case',str(qsms_subs[q]))
            if visualize == True:
                plt.imshow(np.rot90(img),cmap="gray")
                plt.show()
        
        torch.save(X_img,file_path) 
    else:
        X_img = torch.load(file_path)
        subsc = []
        per_change = []
        pre_metric = []
        pre_comp = []
    return X_img, subsc, per_change, pre_metric, pre_comp

def slice_collect(subsc,per_change,pre_metric,pre_comp,pshape,roi_l,roi_u,mask_crop_output,mask_output,o_index,file_path,qsm_path,seg_prefix,L,new_prefix,save_image,img_directory,visualize,reload):
    qsms = full_path(qsm_path)
    writer = sitk.ImageFileWriter()
    qsms_subs = []
    X_img = []
    seg_prefix_dd = seg_prefix[:-1]
    if reload == True:
        for Q in np.arange(len(qsms)):
            qsms_subs.append(int(qsms[Q][-9:-7]))

        for j in np.arange(len(subsc)):
            q = np.where(qsms_subs==subsc[j])[0][0]
            data = nib.load(qsms[q])
            if qsms_subs[q] < L:
                mask = nib.load(seg_prefix+'0'+str(qsms_subs[q])+'.nii.gz').get_fdata()
            elif new_prefix == True:
                mask = nib.load(seg_prefix_dd+str(qsms_subs[q])+'.nii.gz').get_fdata()
            else:
                mask = nib.load(seg_prefix+str(qsms_subs[q])+'.nii.gz').get_fdata()
            mask[mask > roi_u] = 0
            mask[mask < roi_l] = 0
            maskc = pad_to(mask_crop(mask,mask,0),pshape[0],pshape[1],pshape[2])
            mask_k = []
            k_all = []
            if mask_crop_output == True:
                if mask_output == True:
                    data = pad_to(mask_crop(mask*data.get_fdata(),mask,0),pshape[0],pshape[1],pshape[2])
                else:
                    data = pad_to(mask_crop(data.get_fdata(),mask,0),pshape[0],pshape[1],pshape[2])
                for k in np.arange(maskc.shape[2]):
                    if np.sum(maskc[:,:,k]) > 0:
                        mask_k.append(np.sum(maskc[:,:,k]))
                        k_all.append(k)
            else:     
                if o_index == True:
                    data = data.get_fdata()
                    for k in np.arange(mask.shape[2]):
                        if np.sum(mask[:,:,k]) > 0:
                            mask_k.append(np.sum(mask[:,:,k]))
                            k_all.append(k)
                else:
                    data = mask_crop(data.get_fdata(),mask,pshape[0])
                    maskc = mask_crop(mask,mask,pshape[0])
                    for k in np.arange(maskc.shape[2]):
                        if np.sum(maskc[:,:,k]) > 0 and len(np.unique(maskc[:,:,k]))==5:
                            mask_k.append(np.sum(maskc[:,:,k]))
                            k_all.append(k)
                    
            img = data[:,:,k_all]
            if save_image == True:
                for k in np.arange(img.shape[2]):
                    img_sitk =  sitk.Cast(sitk.GetImageFromArray(img),sitk.UInt32)
                    writer.SetFileName(str((img_directory)+'X_img_'+str(k)+'_'+str(j)+'.png'))
                    writer.Execute(img_sitk)
            X_img.append(torch.Tensor(img).cuda())

            if visualize == True:
                plt.imshow(np.rot90(img),cmap="gray")
                plt.show()
        
        torch.save(X_img,file_path) 
    else:
        X_img = torch.load(file_path)
        subsc = []
        per_change = []
        pre_metric = []
        pre_comp = []
    return X_img, subsc, per_change, pre_metric, pre_comp

def filter_data(file_path,cv_names,filter_data):
    df = pd.read_csv(file_path)
    dfd = df.copy()

    if filter_data == True:
        # Drop blank columns
        try:
            for (columnName, columnData) in dfd.iteritems():
                if columnData.isnull().all():
                    print('Dropping NaN column at',columnName)
                    dfd.drop(columnName,axis=1,inplace=True)
            # Add relevant column names from headers
            for (columnName, columnData) in dfd.iteritems():
                    dfd.rename(columns={columnName:columnName+': '+columnData.values[0]},inplace=True)
        except:
            for (columnName, columnData) in dfd.items():
                if columnData.isnull().all():
                    print('Dropping NaN column at',columnName)
                    dfd.drop(columnName,axis=1,inplace=True)
            # Add relevant column names from headers
            for (columnName, columnData) in dfd.items():
                    dfd.rename(columns={columnName:columnName+': '+columnData.values[0]},inplace=True)
        def drop_prefix(self, prefix):
            self.columns = self.columns.str.lstrip(prefix)
            return self
        pd.core.frame.DataFrame.drop_prefix = drop_prefix
        dfd.drop_prefix('Unnamed:')
        motor_df = dfd.copy()      
        try:  
            for (columnName, columnData) in motor_df.iteritems():
                if columnName[1].isdigit():
                    motor_df.rename(columns={columnName:columnName[4:]},inplace=True)
        except:
            for (columnName, columnData) in motor_df.items():
                if columnName[1].isdigit():
                    motor_df.rename(columns={columnName:columnName[4:]},inplace=True)
        # Drop non-motor (III) columns
        for (columnName, columnData) in motor_df.iteritems():
            print(columnName)
            if columnName in cv_names:
                print('Keeping',columnName)
                next
            else:
                motor_df.drop(columnName,axis=1,inplace=True)
        # Drop subheader
        motor_df = motor_df.tail(-1)
        motor_df = motor_df.replace('na',np.nan)
        motor_df = motor_df.dropna()
    else:
        motor_df = dfd
    return motor_df

def filter_scores(file_path,score,key,dose,ids,age,sex,eth,rce,dd,phys,loc,tgt,tms):
    df = pd.read_csv(file_path)
    dfd = df.copy()
    # Drop blank columns
    try:
        for (columnName, columnData) in dfd.iteritems():
            if columnData.isnull().all():
                print('Dropping NaN column at',columnName)
                dfd.drop(columnName,axis=1,inplace=True)
        # Add relevant column names from headers
        for (columnName, columnData) in dfd.iteritems():
                dfd.rename(columns={columnName:columnName+': '+columnData.values[0]},inplace=True)
    except:
        for (columnName, columnData) in dfd.items():
            if columnData.isnull().all():
                print('Dropping NaN column at',columnName)
                dfd.drop(columnName,axis=1,inplace=True)
        # Add relevant column names from headers
        for (columnName, columnData) in dfd.items():
                dfd.rename(columns={columnName:columnName+': '+columnData.values[0]},inplace=True)
    def drop_prefix(self, prefix):
        self.columns = self.columns.str.lstrip(prefix)
        return self
    pd.core.frame.DataFrame.drop_prefix = drop_prefix
    dfd.drop_prefix('Unnamed:')      
    try:  
        for (columnName, columnData) in dfd.iteritems():
            if columnName[1].isdigit():
                dfd.rename(columns={columnName:columnName[4:]},inplace=True)
        # Make a copy for motor symptoms
        motor_df = dfd.copy()
        # Drop non-motor (III) columns
        for (columnName, columnData) in motor_df.iteritems():
            if score in columnName:
                next
            elif key in columnName:
                next
            elif dose in columnName:
                next
            elif ids in columnName:
                next
            elif age in columnName:
                next
            elif sex in columnName:
                next
            elif dd in columnName:
                next
            elif eth in columnName:
                next
            elif rce in columnName:
                next
            elif phys in columnName:
                next
            elif tms in columnName:
                next
            elif loc in columnName:
                next
            elif tgt in columnName:
                next
            else:
                motor_df.drop(columnName,axis=1,inplace=True)
    except:
        for (columnName, columnData) in dfd.items():
            if columnName[1].isdigit():
                dfd.rename(columns={columnName:columnName[4:]},inplace=True)
        # Make a copy for motor symptoms
        motor_df = dfd.copy()
        # Drop non-motor (III) columns
        for (columnName, columnData) in motor_df.items():
            if score in columnName:
                next
            elif key in columnName:
                next            
            elif dose in columnName:
                next
            elif ids in columnName:
                next
            else:
                motor_df.drop(columnName,axis=1,inplace=True)
    # Drop subheader
    motor_df = motor_df.tail(-1)
    motor_df = motor_df.replace('na',np.nan)
    return motor_df

def filter_cases(motor_df,cv_names):
    cv_dict = dict()
    print(cv_names)
    cv_dict = {cv_name: pd.to_numeric(motor_df[cv_name],errors='coerce').to_numpy().astype('float') for cv_name in cv_names}
    return cv_dict

def get_full_cases(df,h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12,h13):
    h0a = pd.to_numeric(df[h0],errors='coerce').to_numpy().astype('float')
    h1a = df[h1].to_numpy().astype('float')
    h2a = df[h2].to_numpy().astype('float')
    h3a = df[h3].to_numpy().astype('float')
    h4a = df[h4].to_numpy().astype('float')
    h5a = df[h5].to_numpy().astype('float')
    h6a = df[h6].to_numpy().astype('float')
    h7a = df[h7].to_numpy().astype('float')
    h8a = df[h8].to_numpy().astype('float')
    h9a = df[h9].to_numpy().astype('float')
    h10a = df[h10].to_numpy().astype('float')
    h11a = df[h11].to_numpy().astype('float')
    h12a = df[h12].to_numpy().astype('float')
    h13a = df[h13].to_numpy().astype('float')
    print(df)
    idx = ~np.isnan(h3a+h2a+h1a+h0a)
    subs = h0a[idx]
    pre_updrs_off = h1a[idx]
    pre_imp = (h1a[idx]-h2a[idx])/h1a[idx]
    true_imp = (h1a[idx]-h3a[idx])/h1a[idx]
    age = h4a[idx]
    sex = h5a[idx]
    eth = h6a[idx]
    rce = h7a[idx]
    dd = h8a[idx]
    phy = h9a[idx]
    ledd = h10a[idx]
    loc = h11a[idx]
    tgt = h12a[idx]
    tms = h13a[idx]
    return subs, pre_imp, true_imp, pre_updrs_off, ledd, age, sex, eth, rce, dd, phy, loc, tgt, tms

def get_fullc_cases(df,h0,h1,h2,h3,h4):
    h0a = pd.to_numeric(df[h0],errors='coerce').to_numpy().astype('float')
    h1a = df[h1].to_numpy().astype('float')
    h2a = df[h2].to_numpy().astype('float')
    h3a = df[h3].to_numpy().astype('float')
    h4a = df[h4].to_numpy().astype('float')

    print(df)
    idx = ~np.isnan(h3a+h2a+h1a+h0a)
    subs = h0a[idx]
    pre_updrs_off = h1a[idx]
    pre_imp = (h1a[idx]-h2a[idx])/h1a[idx]
    true_imp = (h1a[idx]-h3a[idx])/h1a[idx]
    ledd = h4a[idx]
  
    return subs, pre_imp, true_imp, pre_updrs_off, ledd,

def re_index(X_all,K_all,R_all,c_cases_idx,subs,ids,all_rois,pre_imp,pre_updrs_off,post_imp,dose):
    X_all_c = X_all[c_cases_idx,:,:]
    K_all_c = K_all[c_cases_idx,:,:]
    R_all_c = R_all[c_cases_idx,:,:]
    print(np.unique(R_all_c))
    # Re-index the scored subjects with respect to complete cases
    s_cases_idx = np.in1d(subs,ids[c_cases_idx])
    subsc = subs[s_cases_idx]
    pre_imp = pre_imp[s_cases_idx]
    post_imp = post_imp[s_cases_idx]
    pre_updrs_off = pre_updrs_off[s_cases_idx]
    dose = dose[s_cases_idx]
    per_change = (pre_updrs_off-post_imp)/pre_updrs_off
    # Reshape keys and ROIs
    if all_rois == True:
        K_all_cu = np.empty((K_all_c.shape[0],K_all_c.shape[1],K_all_c.shape[2]+1),dtype=object)
        K_all_cu[:,:,:-1] = K_all_c
        K_all_cu[:,:,-1] = 'pre_updrs'
        K = K_all_cu.reshape((K_all_cu.shape[0],K_all_cu.shape[1]*K_all_cu.shape[2]))[0]
        R = R_all_c.reshape((R_all_c.shape[0],R_all_c.shape[1]*R_all_c.shape[2]))
    else:
        K = K_all_c.reshape((K_all_c.shape[0],K_all_c.shape[1]*K_all_c.shape[2]))[0]
        K = np.append(K,['pre_updrs'],0)
        R = R_all_c.reshape((R_all_c.shape[0],R_all_c.shape[1]*R_all_c.shape[2]))
    return X_all_c, K, R, subsc, pre_imp, pre_updrs_off, per_change, dose, s_cases_idx


def fmre_index(X_all,K_all,R_all,c_cases_idx,subs,ids,all_rois,pre_imp,pre_updrs_off,post_imp,dose):
    X_all_c = X_all[c_cases_idx,:,:]
    K_all_c = K_all[c_cases_idx,:,:]
    R_all_c = R_all[c_cases_idx,:,:]
    # Re-index the scored subjects with respect to complete cases
    idsc = ids[np.in1d(ids,subs)]
    idx = np.zeros((1,len(idsc)))
    for j in np.arange(len(idsc)):
        idx[:,j] = int(np.where(idsc[j]==subs)[0][0])
    idx = idx.astype(int)
    subsc = np.squeeze(subs[idx])
    pre_imp =  np.squeeze(pre_imp[idx])
    post_imp =  np.squeeze(post_imp[idx])
    pre_updrs_off =  np.squeeze(pre_updrs_off[idx])
    dose =  np.squeeze(dose[idx])
    per_change = (pre_updrs_off-post_imp)/pre_updrs_off
    # Reshape keys and ROIs
    if all_rois == True:
        K_all_cu = np.empty((K_all_c.shape[0],K_all_c.shape[1],K_all_c.shape[2]+1),dtype=object)
        K_all_cu[:,:,:-1] = K_all_c
        K_all_cu[:,:,-1] = 'pre_updrs'
        K = K_all_cu.reshape((K_all_cu.shape[0],K_all_cu.shape[1]*K_all_cu.shape[2]))[0]
        R = R_all_c.reshape((R_all_c.shape[0],R_all_c.shape[1]*R_all_c.shape[2]))
    else:
        K = K_all_c.reshape((K_all_c.shape[0],K_all_c.shape[1]*K_all_c.shape[2]))[0]
        K = np.append(K,['pre_updrs'],0)
        R = R_all_c.reshape((R_all_c.shape[0],R_all_c.shape[1]*R_all_c.shape[2]))
    return X_all_c, K, R, subsc, pre_imp, pre_updrs_off, per_change, dose, idx

def set_split(X,y,N,tp):
    sss = sms.ShuffleSplit(n_splits=N, test_size=tp)
    sss.get_n_splits(X,y)
    train_index, test_index = next(sss.split(X,y))
    X_train,X_test = X[train_index], X[test_index] 
    y_train,y_test = y[train_index], y[test_index]
    return X_train, X_test, y_train, y_test, train_index, test_index

def make_feature_matrix(X_all_c,idx,pre_metric,dose,age,sex,rce,eth,dd,phy,tgt,tms,scaler,all_roi,slices):
    if all_roi == True:
        X = np.zeros((X_all_c.shape[0],X_all_c.shape[1],X_all_c.shape[2]+2))
        X[:,:,:-1] = X_all_c
        if slices == True:
            X[:,:,-2] = pre_metric[idx]
            X[:,:,-1] = dose[idx]
        else:
            X[:,:,-2] = mb.repmat(pre_metric[idx],X_all_c.shape[1],1).T
            X[:,:,-1] = mb.repmat(dose[idx],X_all_c.shape[1],1).T
        X = X.reshape(X.shape[0],((X.shape[1])*X.shape[2]))
        scaler.fit(X)
        X = scaler.transform(X)
    else:
        X = X_all_c.reshape(X_all_c.shape[0],-1)
        X = np.append(X,pre_metric[idx].reshape(-1,1),1)
        if dose is not None:
            X = np.append(X,dose[idx].reshape(-1,1),1)
        if age is not None:
            X = np.append(X,age[idx].reshape(-1,1),1)
        if sex is not None:
            X = np.append(X,sex[idx].reshape(-1,1),1)
        if rce is not None:
            X = np.append(X,rce[idx].reshape(-1,1),1)
        if eth is not None:
            X = np.append(X,eth[idx].reshape(-1,1),1)
        if dd is not None:
            X = np.append(X,dd[idx].reshape(-1,1),1)
        if phy is not None:
            X = np.append(X,phy[idx].reshape(-1,1),1)
        if tgt is not None:
            X = np.append(X,tgt[idx].reshape(-1,1),1)
        if tms is not None:
            X = np.append(X,tms[idx].reshape(-1,1),1)
        scaler.fit(X)
        X = scaler.transform(X)
    return X, scaler

def model_scale(scaler_type,X_train,train_index,X_test,test_index,pre_metric,dose,age,sex,rce,eth,dd,phy,tgt,tms,all_roi,scale_together,slices):
    scaler = scaler_type
    if scale_together == True:
        X_all = np.vstack((X_train,X_test))
        pre_metric = np.hstack((pre_metric[train_index],pre_metric[test_index]))
        dose = pre_metric = np.hstack((dose[train_index],dose[test_index]))
        X0_tt,scaler = make_feature_matrix(X_all,pre_metric,dose,age,sex,rce,eth,dd,phy,tgt,tms,scaler,all_roi,slices)
        X_test_in = X0_tt[test_index,:]
        X0_tt = X0_tt[train_index,:]
    else:
        X0_tt,scaler = make_feature_matrix(X_train,train_index,pre_metric,dose,age,sex,rce,eth,dd,phy,tgt,tms,scaler,all_roi,slices)
        X_test_in = scale_feature_matrix(X_test,test_index,pre_metric,dose,age,sex,rce,eth,dd,phy,tgt,tms,scaler,all_roi,slices)
    return X0_tt,scaler,X_test_in

def scale_feature_matrix(X_test,idx,pre_metric_test,dose_test,age,sex,rce,eth,dd,phy,tgt,tms,scaler,all_roi,slices):
    if all_roi == True:
        X = np.zeros((X_test.shape[0],X_test.shape[1],X_test.shape[2]+2))
        X[:,:,:-1] = X_test
        if slices == True:
            X[:,:,-2] = pre_metric_test[idx]
            X[:,:,-1] = dose_test[idx]
        else:
            X[:,:,-2] = mb.repmat(pre_metric_test[idx],X_test.shape[1],1).T
            X[:,:,-1] = mb.repmat(dose_test[idx],X_test.shape[1],1).T
        X = X.reshape(X.shape[0],-1)
        X = scaler.transform(X)
    else:
        X = X_test.reshape(X_test.shape[0],-1)
        X = np.append(X,pre_metric_test[idx].reshape(1,-1),1)
        if dose_test is not None:
            X = np.append(X,dose_test[idx].reshape(1,-1),1)
        if age is not None:
            X = np.append(X,age[idx].reshape(1,-1),1)
        if sex is not None:
            X = np.append(X,sex[idx].reshape(1,-1),1)
        if rce is not None:
            X = np.append(X,rce[idx].reshape(1,-1),1)
        if eth is not None:
            X = np.append(X,eth[idx].reshape(1,-1),1)
        if dd is not None:
            X = np.append(X,dd[idx].reshape(1,-1),1)
        if phy is not None:
            X = np.append(X,phy[idx].reshape(1,-1),1)
        if tgt is not None:
            X = np.append(X,tgt[idx].reshape(1,-1),1)
        if tms is not None:
            X = np.append(X,tms[idx].reshape(1,-1),1)
        X = scaler.transform(X)
    return X

def rad_smogn(X_t,y_t,yo1,yu,Rmo,Rmu,t,p,rs):
    warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
    warnings.simplefilter(action='ignore', category=RuntimeWarning)
    # Create data frame for SMOGN generation
    np.random.seed(rs)
    n_cases = len(y_t)
    D = pd.DataFrame(np.hstack((X_t,(np.asarray(y_t).reshape(n_cases,1)))))
    for col in D.columns:
        D.rename(columns={col:str(col)},inplace=True)
    # Specify phi relevance values
    Rm = [[yo1,  Rmo,    0],  
          [yu,   Rmu,    0]]
    
    d = len(D.columns)
    yi = pd.DataFrame(D[str(d-1)])
    # Pre-index targets
    idx = pd.Index((yi.values).ravel())
    # Get sorted indices
    idx = idx.sort_values(return_indexer=True)
    # Sort targets in ascending order
    y_sort = yi.sort_values(by=str(d-1))
    y_sort = y_sort[str(d-1)]
    # Generate relevance function
    phi_params = phi_ctrl_pts(y = y_sort,
        method = 'manual',                                
        ctrl_pts = Rm)
    y_phi = phi(y = y_sort,              
    ctrl_pts = phi_params)
    # Verify sample size reduction
    N_us = np.sum(np.asarray(y_phi)>t)
    idx_kept = (np.asarray(y_phi)<=t)*(idx[1]+1) > 0
    # Conduct SMOGN
    print('Prior to SMOGN sampling, mean is',X_t.mean(),'standard deviation is',X_t.std())
    X_smogn = smogn.smoter(data = D, y = str(D.columns[-1]),
                           rel_method='manual',rel_ctrl_pts_rg = Rm)
    #print('After SMOGN sampling, mean is',X_smogn[:,:-1].mean(),'standard deviation is',X_smogn[:,:-1].std())
    y_smogn = np.asarray(X_smogn)[:,-1]
    X_smogn = np.asarray(X_smogn)[:,:-1]
    sscaler = skp.StandardScaler()
    # X_smogn = sscaler.fit_transform(X_smogn)
    #print('After rescaling, or lackthereof, SMOGN mean is',X_smogn[:,:-1].mean(),'standard deviation is',X_smogn[:,:-1].std())
    return X_smogn,y_smogn,idx_kept,sscaler

def kl_divergence(p, q):
    eps = 1e-6
    p = p/np.sum(p)
    q = q/np.sum(q)
    p = p+eps
    q = q+eps
    kl = np.sum(p*np.log(p/q))
    return kl

def make_pdfs(Q,P,N):
    x = np.arange(np.min((np.min(Q),np.min(P))), np.max((np.max(Q),np.max(P))),1/N)
    p = scipy.stats.norm.pdf(x, np.mean(P), np.std(P))
    q = scipy.stats.norm.pdf(x, np.mean(Q), np.std(Q))
    return p,q

def calc_entropy(s):
    P = [n_x[1]/len(s) for n_x in collections.Counter(s).items()]
    e_x = [-p_x*math.log(p_x,2) for p_x in P]    
    H = sum(e_x)
    return H

def eval_prediction(results,y_test,names,fig_size):
    plt.rcParams["figure.figsize"] = fig_size
    n_models = results.shape[0]
    # Cross validation results
    if np.mod(n_models,2)==0:
        plt.rcParams["figure.figsize"] = (fig_size[0]/2,fig_size[1]*2)
        [fig,ax] = plt.subplots(2,int(n_models//2),sharex=True, sharey=True)
        ax = np.ravel(ax)
        ax_reshape = 2
    elif np.mod(n_models,3)==0:
        plt.rcParams["figure.figsize"] = (fig_size[0]/3,fig_size[1]*3)
        [fig,ax] = plt.subplots(3,int(n_models//3),sharex=True, sharey=True)
        ax = np.ravel(ax)
        ax_reshape = 3
    else:
        [fig,ax] = plt.subplots(1,n_models,sharex=True, sharey=True)
        ax_reshape = 0
    for j in np.arange(n_models):
        lr_prepost = linregress(results[j],y_test)
        ax[j].scatter(results[j],y_test)
        ax[j].plot(results[j],results[j]*lr_prepost.slope+lr_prepost.intercept,'-r')
        ax[j].set_title(names[j])
        ax[j].set_ylabel("DBS improvement")
        ax[j].set_xlabel("Prediction")
        text = f"$y={lr_prepost.slope:0.2f}\; x{lr_prepost.intercept:+0.2f}$\n$r = {lr_prepost.rvalue:0.2f}$\n$p = {lr_prepost.pvalue:0.2e}$"
        ax[j].text(0.35, 0.75, text,transform=ax[j].transAxes,
            fontsize=14, verticalalignment='top')
        ax[j].hlines(0.3,0,2,linestyle='dashed',color='black')
        ax[j].vlines(0.3,0,2,linestyle='dashed',color='black')
    if ax_reshape == 3:
        ax = np.reshape(ax, (3, int(n_models/3)))
    elif ax_reshape == 2:
        ax = np.reshape(ax, (2, int(n_models/2)))
    plt.style.use('default')
    plt.show

def gridsearch_pickparams(model,cvn,param_grid,X0_tt,y_train,scoring,n_js):
    gsc = sms.GridSearchCV(
        model,
        param_grid,
        cv=cvn, 
        scoring=scoring,
        verbose=2,
        n_jobs=n_js,
        return_train_score=True,
        error_score='raise',
        refit=False)
    # X0_tt,scaler,X_test_in = model_scale(scaler_type,X_train,train_index,X_test,test_index,pre_metric)
    grid_result = gsc.fit(X0_tt,y_train)
    best_params = grid_result.best_params_
    return best_params

def find_nearest(array, value):
    array = np.asarray(array) 
    idx = (np.abs(array-value)).argmin()
    return array[idx]

def l_curve(base_min,base_max,X,y,n_points):
    alphas = np.logspace(base_min,base_max,n_points)
    solution_norm = []
    residual_norm = []

    for alpha in alphas: 
        lm = Lasso(alpha=alpha,max_iter=1000,tol=1e-2)
        lm.fit(X, y)
        solution_norm += [(lm.coef_**2).sum()]
        residual_norm += [((lm.predict(X) - y)**2).sum()]

    plt.loglog(residual_norm, solution_norm, 'k-')
    for i, txt in enumerate(alphas):
        if np.mod(i,10)==0:
            plt.annotate(str('{:.2}'.format(10**np.log10(float(txt)).round())), (residual_norm[i], solution_norm[i]+0.005))
    plt.title('L-curve for $(X,y)$')
    plt.xlabel('Residual norm')
    plt.ylabel('Solution norm')
    plt.show()
    plt.style.use('default')

def nii_slicer(case_path,slice_path):
    V = nib.load(case_path).get_fdata()
    for j in np.arange(V.shape[2]):
        Vj = nib.save(V[:,:,j],slice_path)

def roi_var(qsms,segs,roi_keys):
    n_rois = len(roi_keys)
    chi = []
    mask = []
    V = np.zeros((len(qsms),n_rois))
    U = np.zeros((len(qsms),n_rois))
    sub_err = []
    for j in np.arange(len(qsms)):
            mask = nib.load(segs[j]).get_fdata()
            chi = nib.load(qsms[j]).get_fdata()
            for k in np.arange(n_rois):
                if int(qsms[j][-9:-7]) == int(segs[j][-9:-7]):
                    print('Computing variance for case '+qsms[j],'with mask '+segs[j],'and label '+str(roi_keys[k]))
                    M = mask == roi_keys[k]
                    try:
                        V[j,k] = np.var(chi[M==1])
                        U[j,k] = np.mean(chi[M==1])
                    except:
                       sub_err.append(qsms[j][-9:-7])
                       print('Omit variance for case '+qsms[j],'with mask '+segs[j]+'and label '+str(roi_keys[k])) 
    return V, U, sub_err

def pad_to(array,xx,yy,zz):
    h = array.shape[0]
    w = array.shape[1]
    s = array.shape[2]
    a = np.amax(((xx-h)//2,0))
    aa = xx-a-h
    b = np.amax(((yy-w)//2,0))
    bb = yy-b-w
    c = np.amax(((zz-s)//2,0))
    cc = zz-c-s
    # Check for negative values
    aa = np.amax((0,aa))
    bb = np.amax((0,bb))
    cc = np.amax((0,cc))
    return np.pad(array, pad_width=((a, aa), (b, bb), (c, cc)), mode='constant')

def crop_to(array,xx,yy,zz):
    h = array.shape[0]
    w = array.shape[1]
    s = array.shape[2]
    a = h//2-(xx//2)
    b = w//2-(yy//2)
    c = s//2-(zz//2)
    return array[a:a+xx,b:b+yy,c:c+zz]

def encoder(encoder_weights, encoder_biases, data):
    res_ae = data
    for index, (w, b) in enumerate(zip(encoder_weights, encoder_biases)):
        if index+1 == len(encoder_weights):
            res_ae = res_ae@w+b 
        else:
            res_ae = np.maximum(0, res_ae@w+b)
    return res_ae

def full_path(directory):
    return [os.path.join(directory, file) for file in sorted(os.listdir(directory))]

def mask_crop(data,mask,pad):
    if data.shape != mask.shape:
        data = data[:mask.shape[0],:mask.shape[1],:mask.shape[2]]
    img = data[:,:,~(mask==0).all((0,1))]
    img = img[~(mask==0).all((1,2)),:,:]
    img = img[:,~(mask==0).all((0,2)),:]
    if pad != 0:
        x0 = ~(mask==0).all((1,2))
        x00 = np.where(x0)[0][0]
        x0f = np.where(x0)[0][-1]
        y0 = ~(mask==0).all((0,2))
        y00 = np.where(y0)[0][0]
        y0f = np.where(y0)[0][-1]
        z0 = ~(mask==0).all((0,1))
        z00 = np.where(z0)[0][0]
        z0f = np.where(z0)[0][-1]
        img = data[x00:x0f,y00:y0f,z00:z0f]
        x = (pad-img.shape[0])/2
        y = (pad-img.shape[1])/2
        z = (pad-img.shape[2])/2
        img = data[int(x00-x):int(x0f+x),int(y00-y):int(y0f+y),int(z00-z):int(z0f+z)]
    return img

def get_neighbors(y):
    nns = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(y)
    distances, indices = nns.kneighbors(y)
    return indices.flatten()[1::2]

def neighboring_features(X,idx):
    Xn = np.zeros((len(idx),X.shape[1]))
    for j in np.arange(len(idx)):
        Xn[j,:] = X[idx[j],:]
    return torch.Tensor(Xn)

def l1(x,y):
    return torch.norm(x-y,1)

def nstack(a,b,gpu):
    a = np.expand_dims(a,axis=-1)
    while np.ndim(b) < np.ndim(a):
        b = np.expand_dims(b,axis=1)
    if gpu == True:
        c = torch.Tensor(a).cuda()+torch.Tensor(b).cuda()
    else:
        c = a+b
    return c

def confidence_interval(res,pc,xmax):
    lr = stats.linregress(res,pc)
    x_mean = np.mean(res)
    y_mean = np.mean(pc)
    n = len(pc)
    # Slope and intercept parameters                  
    m = 2                           
    dof = n-m
    # Students statistic of interval confidence                    
    t = stats.t.ppf(0.95,dof)      
    residual_ut = pc-np.mean(res,axis=0)
    std_error = (np.sum(residual_ut**2)/dof)**0.5   # Standard deviation of the error
    x = pc
    y = np.mean(res,axis=0)
    # Pearson's correlation coefficient
    numerator = np.sum((x-x_mean)*(y-y_mean))
    denominator = (np.sum((x-x_mean)**2)*np.sum((y-y_mean)**2))**0.5
    correlation_coef = numerator/denominator
    r = correlation_coef
    tr = r*np.sqrt(n-2)/(np.sqrt(1-r**2))

    # to plot the adjusted model
    x_line = np.linspace(0,xmax,100)
    y_line = lr.slope*x_line+lr.intercept
    # confidence interval
    ci = t*std_error*(1/n+(x_line-x_mean)**2/np.sum((x-x_mean)**2))**0.5
    # predicting interval
    pi = t*std_error*(1+1/n+(x_line-x_mean)**2/np.sum((x-x_mean)**2))**0.5  

    return ci, pi, x_line, y_line

def get_layer_output(x_img,model,layer):
    modelz = create_feature_extractor(model, return_nodes=layer)
    z = modelz(torch.unsqueeze(torch.repeat_interleave(torch.unsqueeze(x_img.cpu(),axis=0), 3, dim=0),axis=0))
    return z

def get_latent_rep(x_img,model):
    print(model)
    z = model(torch.unsqueeze(torch.repeat_interleave(torch.unsqueeze(x_img.cpu(),axis=0), 3, dim=0),axis=0))
    return z

def conv_2d_layer_dim(in_size,layer):
    hi,wi = in_size
    p = layer.padding
    d = layer.dilation
    k = layer.kernel_size
    s = layer.stride
    ho = ((hi+2*p-d*(k-1)-1)/s)+1
    wo = ((wi+2*p-d*(k-1)-1)/s)+1
    return ho,wo

def focal_loss(output,target,beta,gamma):
    sigmoid = nn.Sigmoid()
    ei = torch.norm(target-output,1)
    loss = torch.mean(sigmoid((torch.abs(beta*ei,1)**gamma)*ei))

def classification_accuracy(y_pred,y_true):
    acc = np.sum(y_pred==y_true)/len(y_true)
    return acc

def classification_specificity(y_pred,y_true):
    tn = np.sum((y_pred==0)==(y_true==0))
    fp = np.sum(np.logical_and((y_pred==1).astype(int),(y_true==0).astype(int)))
    spf = tn/(tn+fp)
    return spf

def classification_sensitivity(y_pred,y_true):
    tp = np.sum((y_pred==1)==(y_true==1))
    fn = np.sum(np.logical_and((y_pred==0).astype(int),(y_pred==1).astype(int)))
    sns = tp/(tp+fn)
    return sns

def make_feature_map(subject,feat_name,mask,rois,w,l,cmin,cmax,normalize):

    # Import the maps from `.nrrd` files
    r_num1 = rois[0]
    r_num2 = rois[1]
    pr = subject

    # ROI 1
    hr1,_ = nrrd.read('./'+feat_name+'_'+str(pr)+'_'+str(r_num1)+'.nrrd')
    if np.sum(mask) > 0:
        hrq1,_ = nrrd.read('./maps/msw_qsm_crop_'+str(pr)+'_'+str(r_num1)+'.nrrd')
        mr1,mrq1 = nrrd.read('./maps/msw_seg_crop_'+str(pr)+'_'+str(r_num1)+'.nrrd')
        hr1 = hrq1*(abs(hr1)!=0)
    
    # ROI 2
    hr2,_ = nrrd.read('./'+feat_name+'_'+str(pr)+'_'+str(r_num2)+'.nrrd')
    if np.sum(mask) > 0:
        hrq2,_ = nrrd.read('./maps/msw_qsm_crop_'+str(pr)+'_'+str(r_num2)+'.nrrd')
        mr2,mrq2 = nrrd.read('./maps/msw_seg_crop_'+str(pr)+'_'+str(r_num2)+'.nrrd')
        hr2 = hrq2*(abs(hr2)!=0)

    # Pad the ROIs to the same shape
    px,py,pz = np.asarray(hr1.shape)-np.asarray(hr2.shape)
    hr1p = np.pad(hr1,((0,abs(min(0,px))),(0,abs(min(0,py))),(0,abs(min(0,pz)))))
    hr2p = np.pad(hr2,((0,abs(max(0,px))),(0,abs(max(0,py))),(0,abs(max(0,pz)))))
    hr = np.flipud(np.concatenate((hr1p,hr2p),axis=2))

    if normalize == True:
        hr = hr/np.amax(hr)
    
    if np.sum(mask) > 0:
        pyvis(mask*hr,10,10,'gray',w,l,cmin,cmax)
    else:
        pyvis(hr,10,10,'gray',w,l,cmin,cmax)

def nrrd_read(file):
    hr,_ = nrrd.read(file)
    pyvis(hr,10,10,'gray',0,0,0,0)

def feature_map_mask(subject,feat_name,pq,rois):

    # Import the maps from `.nrrd` files
    r_num1 = rois[0]
    r_num2 = rois[1]
    pr = subject

    # ROI 1
    hr1,_ = nrrd.read('./'+feat_name+'_'+str(pr)+'_'+str(r_num1)+'.nrrd')
    
    # ROI 2
    hr2,_ = nrrd.read('./'+feat_name+'_'+str(pr)+'_'+str(r_num2)+'.nrrd')

    # Pad the ROIs to the same shape
    px,py,pz = np.asarray(hr1.shape)-np.asarray(hr2.shape)
    hr1p = np.pad(hr1,((0,abs(min(0,px))),(0,abs(min(0,py))),(0,abs(min(0,pz)))))
    hr2p = np.pad(hr2,((0,abs(max(0,px))),(0,abs(max(0,py))),(0,abs(max(0,pz)))))
    hr = np.flipud(np.concatenate((hr1p,hr2p),axis=2))
    mask = abs(hr) != 0
    return mask

def nominal_encode(X,category,categories,verbose):
    # Create nominal rater encoding and append it
    N = len(np.unique(categories))
    x = np.zeros((X.shape[0],N))
    X_cat = np.zeros((X.shape[0],X.shape[1]+N))
    if verbose == 1:
        print('Initializing zero array of shape:',x.shape)
        print('Will append to feature matrix of shape:',X.shape)
        print('Encoding nominal features of shape:',category.shape)
        print('With number of unique values:',N)
        print('Creating new feature matrix of shape:',X_cat.shape)
    X_cat[:,0:X.shape[1]] = X
    for jj in np.arange(X.shape[0]):
        phys_idx = int(category[jj])
        x[jj,phys_idx] = 1
    
    X_cat[:,-N:] = x
    if verbose == 1:
        print('Appending',x,'for category',category,'in categories',np.unique(categories))
    return X_cat