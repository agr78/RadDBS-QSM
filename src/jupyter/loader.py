import nibabel as nib
import os
import pickle
import numpy as np

def data_loader(q_directory,s_directory,reload,suffix,qsm_prefix):
    # Set window level

    # Load data
    segs = []
    qsms = []
    voxel_sizes = []
    s_dir = s_directory
    q_dir = q_directory
    s_directory = os.listdir(s_directory)
    s_directory = sorted(s_directory)

    q_directory = os.listdir(q_directory)
    q_directory = sorted(q_directory)

    case_list = []
    d_count = 0
    if reload == 1:
        for seg_filename in s_directory:
            id = seg_filename[12:14]
            if os.path.isfile(s_dir+seg_filename) and os.path.isfile(q_dir+qsm_prefix+str(id)+'.nii.gz'):
                seg = nib.load(s_dir+seg_filename)
                voxel_size = seg.header['pixdim'][0:3]
                voxel_sizes.append(voxel_size)
                segs.append(seg.get_fdata())
                qsm = nib.load(q_dir+qsm_prefix+str(id)+'.nii.gz')
                qsms.append(qsm.get_fdata())
                print('Appending arrays with segmentation',seg_filename,'and QSM',str(id)+'.nii.gz')
                case_list.append('qsm_'+str(id)+'.nii.gz')
                n_cases = len(segs)
                d_count = d_count+1
                with open('./pickles/segs_'+suffix, 'wb') as fp:  
                    pickle.dump(segs, fp)

                with open('./pickles/qsms_'+suffix, 'wb') as fp:  
                    pickle.dump(qsms, fp)
                
                with open('./pickles/qsms_'+suffix, 'wb') as fp:  
                    pickle.dump(case_list, fp)
            else:
                print('Skipping',seg_filename,'and QSM',str(id))

    else:
        with open('/data/Ali/RadDBS-QSM/src/jupyter/pickles/segs_'+suffix, "rb") as fp:  
            segs = pickle.load(fp)
            n_cases = len(segs)
        with open('/data/Ali/RadDBS-QSM/src/jupyter/pickles/qsms_'+suffix, "rb") as fp:  
            qsms = pickle.load(fp)
        with open('/data/Ali/RadDBS-QSM/src/jupyter/pickles/cases_'+suffix, "rb") as fp:  
            try:
                case_list = pickle.load(fp)
            except:
                case_list = open('/data/Ali/RadDBS-QSM/src/jupyter/pickles/cases_'+suffix,'r')
    return segs, qsms, n_cases, case_list