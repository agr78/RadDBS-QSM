import nibabel as nib
import os
import pickle

def data_loader(q_directory,s_directory,reload):
    # Set window level
    level = 0
    window = 500
    m1=level-window/2
    m2=level+window/2
    visualize = 1
    reextract = 0
    reload = 0
    # Load data
    nrows = 256
    ncols = 256

    nslices = 160
    segs = []
    qsms = []
    laros = []
    voxel_sizes = []
    trackers = []
    q_directory = '/media/mts_dbs/dbs/complete_cases/nii/qsm/'
    s_directory = '/media/mts_dbs/dbs/complete_cases/nii/seg/'
    s_directory = os.listdir(s_directory)
    s_directory = sorted(s_directory)

    case_list = []
    d_count = 0
    if reload == 1:
        for seg_filename in s_directory:
            id = seg_filename[12:14]
            seg = nib.load('/media/mts_dbs/dbs/complete_cases/nii/seg/'+seg_filename)
            voxel_size = seg.header['pixdim'][0:3]
            voxel_sizes.append(voxel_size)
            segs.append(seg.get_fdata())
            qsm = nib.load('/media/mts_dbs/dbs/complete_cases/nii/qsm/qsm_'+str(id)+'.nii.gz')
            qsms.append(qsm.get_fdata())
            print('Appending arrays with segmentation',seg_filename,'and QSM','qsm_'+str(id)+'.nii.gz')
            case_list.append('qsm_'+str(id)+'.nii.gz')
            n_cases = len(segs)
            d_count = d_count+1
            qsms_wl = np.asarray(qsms)
            segs_wl = np.asarray(segs)
            with open('./pickles/segs', 'wb') as fp:  
                pickle.dump(segs, fp)

            with open('./pickles/qsms', 'wb') as fp:  
                pickle.dump(qsms, fp)

    else:
        with open('/data/Ali/RadDBS-QSM/src/jupyter/pickles/segs', "rb") as fp:  
            segs = pickle.load(fp)

        with open('/data/Ali/RadDBS-QSM/src/jupyter/pickles/qsms', "rb") as fp:  
            qsms = pickle.load(fp)