import IPython
from pathlib import Path
import os
import nibabel as nib
import numpy as np

def get_raddbs_path(notebook_name):
    raddbs_path = notebook_name[:notebook_name.index("src")]
    return raddbs_path

def s1data(raddbs_path,nii_dir,verbose):
    # Load data
    nrows = 256
    ncols = 256
    nslices = 160
    segs = []
    qsms = []
    voxel_sizes = []

    trackers = []
    q_directory = Path(raddbs_path+nii_dir+'qsm')
    q_directory = os.listdir(q_directory)
    q_directory = sorted(q_directory)

    s_directory = Path(raddbs_path+nii_dir+'seg')
    s_directory = os.listdir(s_directory)
    s_directory = sorted(s_directory)
    m_directory = Path(raddbs_path+nii_dir+'masks')
    m_directory = os.listdir(m_directory)
    m_directory = sorted(m_directory)
    case_list = []
    d_count = 0

    #if reload == 1:
    # for seg_filename in s_directory:
    #     id = seg_filename[12:14]

    # with open('/home/ali/RadDBS-QSM/src/jupyter/pickles/qsms_'+suffix, "rb") as fp:  
    #     qsms = pickle.load(fp)
    # with open('/home/ali/RadDBS-QSM/src/jupyter/pickles/cases_'+suffix, "rb") as fp:  
    #         cases = pickle.load(fp)

    for filename in q_directory:
        seg_filename = s_directory[d_count]
        mask_filename = m_directory[d_count]
        seg = nib.load(Path(raddbs_path+nii_dir+'/seg/'+seg_filename))
        mask = nib.load(Path(raddbs_path+nii_dir+'masks/'+mask_filename))
        voxel_size = seg.header['pixdim'][0:3]
        voxel_sizes.append(voxel_size)

        ## OUT
        segs.append(seg.get_fdata()[:nrows,:ncols,:nslices])
        
        qsm = nib.load(Path(raddbs_path+'/data/nii/chh/orig/qsm/'+filename))
        qsms.append(qsm.get_fdata()[:nrows,:ncols,:nslices])
        if verbose is True:
            print('Appending arrays with segmentation',seg_filename,'QSM,',filename,'and mask',mask_filename)
        case_list.append(filename)

        ## OUT
        n_cases = len(segs)

        d_count = d_count+1
    return segs, qsms, n_cases

def s1cvdata(df,segs,raddbs_path,nii_dir,verbose):
    # Patient IDs
    try:
        n_cases = len(segs)
        subject_id = np.asarray(df[df.columns[0]])[1:]
        s_directory = Path(raddbs_path+nii_dir+'seg')
        s_directory = os.listdir(s_directory)
        s_directory = sorted(s_directory)

        # Only extract ROI if it is present in all cases
        seg_labels_all = segs[0]
        case_number = np.zeros_like(np.asarray(s_directory))
        for i in range(n_cases):
            case_number[i] = float(s_directory[i][:2])
        subject_id_corr_mask = np.in1d(subject_id,case_number)
        subject_id_corr = subject_id[subject_id_corr_mask]
    except:
        print('Using demo case IDs')
        subject_id_corr = np.asarray([1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 13., 14.,
       16., 18., 19., 20., 21., 22., 23., 24., 25., 26., 27., 28., 29.,
       30., 31., 32., 33., 34., 35., 36., 37., 38., 39., 40.])
        subject_id_corr_mask = np.in1d(subject_id,subject_id_corr)
    age = np.nan_to_num(np.asarray(df[df.columns[-4]])[1:][subject_id_corr_mask].astype(float))
    sex = np.nan_to_num(np.asarray(df[df.columns[-3]])[1:][subject_id_corr_mask].astype(float))
    dd = np.nan_to_num(np.asarray(df[df.columns[-2]])[1:][subject_id_corr_mask].astype(float))
    ledd = np.nan_to_num(np.asarray(df[df.columns[-1]])[1:][subject_id_corr_mask].astype(float))

    if verbose is True:
        for i in np.arange(n_cases):
            try:
                print('Found ROIs',str(np.unique(segs[i])),'at segmentation directory file',s_directory[i],'for case',str(subject_id_corr[i]))
            except:
                case_list = open('/home/ali/RadDBS-QSM/src/jupyter/pickles/cases_'+suffix,'r')
                print('Case',subject_id[i],'quarantined')

    pre_updrs_off =  np.asarray(df[df.columns[3]][np.hstack((False,subject_id_corr_mask))])                                
    pre_updrs_on =  np.asarray(df[df.columns[4]][np.hstack((False,subject_id_corr_mask))])
    post_updrs_off =  np.asarray(df[df.columns[6]][np.hstack((False,subject_id_corr_mask))])

    per_change = (np.asarray(pre_updrs_off).astype(float)-np.asarray(post_updrs_off).astype(float))/(np.asarray(pre_updrs_off).astype(float))
    lct_change = (np.asarray(pre_updrs_off).astype(float)-(np.asarray(pre_updrs_on)).astype(float))/(np.asarray(pre_updrs_off).astype(float))
    subsc = subject_id_corr
    return age,sex,dd,ledd,subsc,pre_updrs_off,pre_updrs_on,post_updrs_off,per_change,lct_change