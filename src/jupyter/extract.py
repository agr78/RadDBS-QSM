import logging
import numpy as np
from pathlib import Path
import pickle
from datetime import datetime
import SimpleITK as sitk
import six
from radiomics import featureextractor 

# n_cases = len(per_change)
# L = int(len(X0_gt)/n_cases)
# n_features = int(L/n_rois)

def feature_matrix(raddbs_path,segs,qsms,subsc,reextract):
    logger = logging.getLogger("radiomics")
    logger.setLevel(logging.ERROR)
    # Assume all voxel sizes are identical
    voxel_size = (0.9,0.9,0.9)
    # Get the current date and time
    now = datetime.now()
    cdt = now.strftime("%Y%m%d%H%M%S")
    # Get the number of cases
    n_cases = len(segs)
    if reextract == 1:
        # Generate feature structure Phi from all ROIs and all cases
        extractor = featureextractor.RadiomicsFeatureExtractor()
        extractor.enableAllFeatures()
        extractor.enableAllImageTypes()
        extractor.enableFeatureClassByName('shape2D',enabled = False)

        seg_labels_all = np.unique(np.asarray(segs))
        Phi_gt = []
        Phi_vd = []
        Phi_lr = []
        seg_labels = []
        x_row_gt = []
        x_row_lr = []

        keylib = []
        roilib = []
        loop_count = 1
        n_rois = seg_labels_all[seg_labels_all>0].__len__()
        roi_names = np.asarray(['Background','Right substantia nigra','Right subthalamic nucleus',
                                'Left subthalamic nucleus', 'Left substantia nigra', 'Right dentate nucleus', 'Left dentate nucleus'])
        for i in np.arange(subsc.__len__()):
            seg_sitk = sitk.GetImageFromArray(segs[i])
            seg_sitk.SetSpacing(voxel_size)
            qsm_sitk_gt = sitk.GetImageFromArray(qsms[i])
            qsm_sitk_gt.SetSpacing(voxel_size)
            # Index back since subject 12 is missing ROIs
            for j in seg_labels_all:
                if j>0:
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
                            roilib.append(roi_names[int(j)])
                    #x_row_gt.append(pre_updrs_iii_off[i])
                    fv_count = fv_count+1
            print('Extracting features for subject',subsc[i],'and appending feature matrix with vector of length',fv_count)#,'with UPDRS score',pre_updrs_iii_off[i])
                    
        X0_gt = np.array(x_row_gt)
        np.save(raddbs_path+'/data/npy/rp/X0_gt_chh_rois_rp_'+cdt+'_.npy',X0_gt)

        K = np.asarray(keylib)
        R = np.asarray(roi_names)
        np.save(raddbs_path+'/data/npy/rp/K_chh_rp_'+cdt+'.npy',K)
        np.save(raddbs_path+'/data/npy/rp/R_chh_rp_'+cdt+'.npy',R)

        print('Saving ground truth feature vector')
        with open(raddbs_path+'/data/npy/rp/Phi_mcl_gt_'+cdt+'.npy', 'wb') as fp:  
            pickle.dump(Phi_gt, fp)

    else:
        X0_gt = np.load(Path(raddbs_path+'/data/npy/rp/X0_gt_chh_rois_rp.npy'))
        K = np.load(Path(raddbs_path+'/data/npy/rp/K_chh_rp.npy'))
        R = np.load(Path(raddbs_path+'/data/npy/rp/R_chh_rp.npy'))
        n_rois = R.shape[0]-1
        with open(Path(raddbs_path+'/data/npy/rp/Phi_mcl_gt_roi_chh_rp'), "rb") as fp:  
            Phi_gt = pickle.load(fp)
        

    n_features = 1596
    n_rois = 4
    X_all_c = X0_gt.reshape(n_cases,n_rois,n_features)[:,0:4,:]
    K_all_c = K.reshape(n_cases,n_rois,n_features-1)[:,0:4,:]
    K_all_c = np.char.add(K_all_c[0,:,:].reshape(-1,1),' ')
    R_all_c = np.repeat(R[1:5],n_features-1)
    K_all_c = np.char.add(np.squeeze(K_all_c),np.squeeze(R_all_c))
    # K_all_c = np.append(K_all_c,['pre updrs']*5)
    # R_all_c = np.append(R_all_c,['pre updrs']*5)
    # K_all_c = np.append(K_all_c,['age'])
    # R_all_c = np.append(R_all_c,['age'])
    # K_all_c = np.append(K_all_c,['disease duration'])
    # R_all_c = np.append(R_all_c,['disease duration'])
    # K_all_c = np.append(K_all_c,['sex'])
    # R_all_c = np.append(R_all_c,['sex'])

    return X_all_c,K_all_c,R_all_c