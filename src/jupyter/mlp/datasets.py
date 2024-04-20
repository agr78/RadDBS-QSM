import sys
sys.path.append('../')
import numpy as np
import nibabel as nib
import numpy as np
import torch
import torch.utils.data as data
import torchio as tio
from imbalanced_regression.utils import get_lds_kernel_window
import logging
from scipy.ndimage import convolve1d
from torch.utils import data
import torchio.transforms as transforms
from imbalanced_regression.qsm.utils import pyvis, model_scale, scale_feature_matrix
import sklearn.preprocessing as skp
import util
import matplotlib.pyplot as plt


class QSM_slices(data.Dataset):
    def __init__(self, data_dir, subfolder, aug_state, factor, X, subsc, targets, prefix):
        if prefix is None:
            if aug_state == True:
                self.images_list = [np.load(subfolder+str(factor)+'/'+image_path) for image_path in data_dir]
            else:
                self.images_list = [np.load('./slices/'+image_path) for image_path in data_dir]
        else:
            if aug_state == True:
                self.images_list = [np.load(prefix+subfolder+str(factor)+'/'+image_path) for image_path in data_dir]
            else:
                self.images_list = [np.load(prefix+'/slices/'+image_path) for image_path in data_dir]
        self.data_dir = data_dir
        self.X = X
        self.subsc = subsc
        self.targets = targets

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, index):
        # print('New case for index:',index)
        case_dir = self.data_dir[index]
        # print('Case path',case_dir)
        # print('Data directory',self.data_dir)
        # print('Loading image',case_dir,'with original index',str(index))
        # print('ID with original index',self.subsc[index])
        nii_image = self.images_list[index]
        data = nii_image
        img = torch.Tensor(data)
        self.img_size = img.shape
        case_int = int(float(case_dir.split("_")[1]))
        # print('Case integer:',case_int,'from case path',case_dir)
        # print('Integer value:',str(case_int))
        # print('Subject array',self.subsc)
        # print('Case integer',case_int)
        # print('Finding the right index',np.where((self.subsc==case_int)))
        # print('Search result:',np.where((self.subsc==case_int)))
        index = np.where((self.subsc==case_int))[0][0]
    
        # print('Index:',index)
        # print('Where ID and file match:',index)
        # print('ID with new index:',self.subsc[index])
        target = self.targets[index]
        X = torch.Tensor(self.X[index,:])
        transform = tio.ZNormalization()
        #transform = self.get_transform(img)
        if img.dim() < 3:
            next
        else:
            img = torch.squeeze(img)
        img = transform(torch.unsqueeze(torch.unsqueeze(img,dim=0),dim=-1))
        # print('Image shape:',img.shape)
        # print('Features shape:',X.shape)
        #img = transform(torch.unsqueeze(torch.unsqueeze(img,axis=-1),axis=0)).reshape(1,self.img_size[0],self.img_size[1])
        label = torch.Tensor(target)
        # print('Label shape:',label.shape)
        # print('Image shape:',img.shape)
        # print('Radiomic feature shapes:',X.shape)
        # print('__getitem__() complete')
        return img, X, label

    def get_transform(self,img):
        self.img = img
        self.img_size = (self.img).shape
        transform=transforms.RandomAffine()
        return transform
    
    def augment_slices(self,img):
        self.img = img
        aug_img = self.get_transform(img)
        return aug_img


def make_slices(data_path,subs,per_change,pre_updrs_off,ledd,X):
    qsms = util.full_path(data_path)
    qsms_subs = []
    subsc = subs
    per_change_out = per_change
    pre_updrs_off_out = pre_updrs_off
    ledd_out = ledd
    X_out = X
    for Q in np.arange(len(qsms)):
        qsms_subs.append(int(qsms[Q][-9:-7]))

    for j in np.arange(len(subs)):
        q = np.where(qsms_subs==subs[j])[0][0]
        data = nib.load(qsms[q])
        # try:
        if qsms_subs[q] < 10:
            mask = nib.load('/home/ali/RadDBS-QSM/data/nii/seg/labels_2iMag0'+str(qsms_subs[q])+'.nii.gz').get_fdata()
        else:
            mask = nib.load('/home/ali/RadDBS-QSM/data/nii/seg/labels_2iMag'+str(qsms_subs[q])+'.nii.gz').get_fdata()
        mask[mask > 4] = 0
        mask[mask < 3] = 0
        maskc = util.crop_to(util.pad_to(util.mask_crop(mask,mask,0),64,64,64),64,64,64)
        data = util.crop_to(util.pad_to(util.mask_crop(mask*data.get_fdata(),mask,0),64,64,64),64,64,64)
        mask_k = []
        k_all = []
        for k in np.arange(maskc.shape[2]):
                if np.sum(maskc[:,:,k]) > 0: #and len(np.unique(maskc[:,:,k]))==len(np.unique(mask)):
                    mask_k.append(np.sum(maskc[:,:,k]))
                    k_all.append(k)
            
        img = data[:,:,k_all[np.argmax(mask_k)]]
        np.save('./slices/case_'+str(subs[j])+'.npy',img)
        print('Maximum volume found at slice',str(k_all[np.argmax(mask_k)]),'for case',str(qsms_subs[q]),'with ID',str(subs[j]),'and shape',str(img.shape))
        plt.imshow(img)
        plt.show()
        # except:
        #     print('Missing mask at',str(qsms_subs[q]))
        #     subsc = np.delete(subs,j)
        #     per_change_out = np.delete(per_change,j)
        #     pre_updrs_off_out = np.delete(pre_updrs_off,j)
        #     ledd_out = np.delete(ledd,j)
        #     X_out = np.delete(X,j,axis=0)
    return subsc,per_change_out,pre_updrs_off_out,ledd_out,X_out

def augment_dataset(imgs,factor,y,subs,Xr):
        torch.manual_seed(0)
        N = len(imgs)
        aug_img = []
        y_aug = []
        subs_aug = []
        Xr_aug = []
        for j in np.arange(N):
            for k in np.arange(factor):
                transform=transforms.RandomAffine(scales=0.05,degrees=10,image_interpolation='nearest')
                aug_img.append(transform(torch.unsqueeze(torch.unsqueeze(torch.squeeze(imgs[j]),dim=0),dim=-1)))
                y_aug.append(y[j])
                subs_aug.append(subs[j])
                Xr_aug.append(Xr[j])
        return aug_img, y_aug, subs_aug, Xr_aug

def elastic_augment_dataset(imgs,factor,y,subs,Xr):
        torch.manual_seed(0)
        N = len(imgs)
        aug_img = []
        y_aug = []
        subs_aug = []
        Xr_aug = []
        for j in np.arange(N):
            for k in np.arange(factor):
                transforms_dict = {transforms.RandomAffine(scales=0.05,degrees=10,image_interpolation='nearest'): 0.5,
                                 transforms.RandomElasticDeformation(image_interpolation='nearest'): 0.5}
                transform = tio.OneOf(transforms_dict)
                aug_img.append(transform(torch.unsqueeze(torch.unsqueeze(torch.squeeze(imgs[j].cpu()),dim=0),dim=-1)))
                y_aug.append(y[j])
                subs_aug.append(subs[j])
                Xr_aug.append(Xr[j])
        return aug_img, y_aug, subs_aug, Xr_aug

class QSM(data.Dataset):
    def __init__(self, data_dir, mask_dir, targets, nz, nx, split='train', reweight='none',
                 lds=False, lds_kernel='gaussian', lds_ks=5, lds_sigma=2):
        self.images_list = [nib.load(image_path) for image_path in data_dir]
        self.masks_list = [nib.load(mask_path) for mask_path in mask_dir]
        self.data_dir = data_dir
        self.mask_dir = mask_dir
        self.targets = targets
        self.nz = nz
        self.nx = nx
        self.split = split
        self.weights = self._prepare_weights(reweight=reweight, lds=lds, lds_kernel=lds_kernel, lds_ks=lds_ks, lds_sigma=lds_sigma)

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, index):
        case_dir = self.data_dir[index]
        #print(case_dir)
        nx = self.nx
        nz = self.nz
        #print('nz is ',nz, ' and nx is ',nx)
        nii_image = self.images_list[index]
        nii_mask = self.masks_list[index]
        data = np.asarray(nii_image.dataobj)
        mask = np.asarray(nii_mask.dataobj)
        #print('Applying mask of shape ',str(mask.shape),' to image of size ',str(data.shape),' for ',case_dir)#,' with size ',str(self.img_size)+' before transform')
        #try:
        img = torch.from_numpy(data[~(mask==0).all(1,2),~(mask==0).all(0,2),~(mask==0).all((0,1))])
        #except:
        #    print(case_dir,' mask has shape ',str(mask.shape))
        self.img_size = img.shape
        target = self.targets[index]
        transform = self.get_transform(img,nx,nz)
        img = torch.squeeze(transform(torch.unsqueeze(img,axis=0)))
        #print(case_dir+' has size ',str(img.shape)+' after transform')
        label = target
        weight = np.asarray([self.weights[index]]).astype('float32') if self.weights is not None else np.asarray([np.float32(1.)])

        return img, label, weight

    def get_transform(self,img,nx,nz):
        self.img = img
        self.img_size = (self.img).shape
        if self.img_size[2]>nz:
            self.img = self.img[:,:,(self.img_size[2]//2)-(nz//2):(self.img_size[2]//2)+(nz//2)]
            tpad = transforms.Pad((0,0,0))
        else:
            if (nz-self.img_size[2])/2 == (nz-self.img_size[2])//2:
                tpad = transforms.Pad((0,0,(nz-self.img_size[2])//2))
            else:
                #print('Padding an odd number of slices with ',str((nz-self.img_size[2])//2),' and ',str(((nz-self.img_size[2])//2)+1))                      
                tpad = transforms.Pad((0,0,0,0,
                                    (nz-self.img_size[2])//2,
                                    ((nz-self.img_size[2])//2)+1))
        if self.split == 'train':
            transform = transforms.Compose([
                transforms.Crop((nx,nx,nx,nx,0,0)),
                tpad,
                #transforms.RandomFlip(axes=['LR', 'AP', 'IS']),
                transforms.RescaleIntensity(out_min_max=(-1, 1)),
            ])
        else:
            transform = transforms.Compose([
                transforms.Crop((nx,nx,nx,nx,0,0)),
                tpad,
                transforms.RescaleIntensity(out_min_max=(-1, 1)),
            ])
        return transform

    def _prepare_weights(self, reweight, max_target=1, lds=False, lds_kernel='gaussian', lds_ks=5, lds_sigma=2):
        assert reweight in {'none', 'inverse', 'sqrt_inv'}
        assert reweight != 'none' if lds else True, \
            "Set reweight to \'sqrt_inv\' (default) or \'inverse\' when using LDS"

        value_dict = {x: 0 for x in range(max_target)}
        labels = self.targets
        for label in labels:
            value_dict[min(max_target - 1, int(label))] += 1
        if reweight == 'sqrt_inv':
            value_dict = {k: np.sqrt(v) for k, v in value_dict.items()}
        elif reweight == 'inverse':
            value_dict = {k: np.clip(v, 5, 1000) for k, v in value_dict.items()}  # clip weights for inverse re-weight
        num_per_label = [value_dict[min(max_target - 1, int(label))] for label in labels]
        if not len(num_per_label) or reweight == 'none':
            return None
        print(f"Using re-weighting: [{reweight.upper()}]")

        if lds:
            lds_kernel_window = get_lds_kernel_window(lds_kernel, lds_ks, lds_sigma)
            print(f'Using LDS: [{lds_kernel.upper()}] ({lds_ks}/{lds_sigma})')
            smoothed_value = convolve1d(
                np.asarray([v for _, v in value_dict.items()]), weights=lds_kernel_window, mode='constant')
            num_per_label = [smoothed_value[min(max_target - 1, int(label))] for label in labels]

        weights = [np.float32(1 / x) for x in num_per_label]
        scaling = len(weights) / np.sum(weights)
        weights = [scaling * x for x in weights]
        return weights


class QSM_features(data.Dataset):
    def __init__(self, data_dir, targets, pre_metric, scaler_ss=None, split='train', reweight='none',
                 lds=False, lds_kernel='gaussian', lds_ks=5, lds_sigma=2):
        self.feature_matrices = [np.load(feature_matrix) for feature_matrix in data_dir]
        self.data_dir = data_dir
        self.targets = targets
        self.split = split
        self.pre_updrs_off = pre_metric
        self.scaler_ss = scaler_ss
        self.weights = self._prepare_weights(reweight=reweight, lds=lds, lds_kernel=lds_kernel, lds_ks=lds_ks, lds_sigma=lds_sigma)
        if self.split == 'train':
            self.data,self.scaler_ss = model_scale(skp.MinMaxScaler(),
                                                np.asarray(self.feature_matrices[:]).reshape((len(targets),6,1595)),self.pre_updrs_off)
        else:
            self.data = (scale_feature_matrix(np.asarray(self.feature_matrices[:]).reshape((len(targets),6,1595)),pre_metric,self.scaler_ss)).reshape(len(targets),6*1596)
    
    def __len__(self):
        return len(self.feature_matrices)

    def __getitem__(self, index):
        case_dir = self.data_dir[index]
        X0 = self.data[index]  
        img = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(np.float32(X0)),dim=1),dim=2)
        label = self.targets[index]
        weight = np.asarray([self.weights[index]]).astype('float32') if self.weights is not None else np.asarray([np.float32(1.)])
        return img, label, weight

    def __getscaler__(self):
        return(self.scaler_ss)
    
    def _prepare_weights(self, reweight, max_target=1, lds=False, lds_kernel='gaussian', lds_ks=5, lds_sigma=2):
        assert reweight in {'none', 'inverse', 'sqrt_inv'}
        assert reweight != 'none' if lds else True, \
            "Set reweight to \'sqrt_inv\' (default) or \'inverse\' when using LDS"
        value_dict =  {x: 0 for x in np.round(np.linspace(0,max_target,11),1)}
        labels = self.targets
        for label in labels:
            value_dict[min(max_target, label)] += 1
        if reweight == 'sqrt_inv':
            value_dict = {k: np.sqrt(v) for k, v in value_dict.items()}
        elif reweight == 'inverse':
            value_dict = {k: np.clip(v, 5, 1000) for k, v in value_dict.items()}  # clip weights for inverse re-weight
        num_per_label = [value_dict[min(max_target - 1, int(label))] for label in labels]
        if not len(num_per_label) or reweight == 'none':
            return None
        print(f"Using re-weighting: [{reweight.upper()}]")

        if lds:
            lds_kernel_window = get_lds_kernel_window(lds_kernel, lds_ks, lds_sigma)
            print(f'Using LDS: [{lds_kernel.upper()}] ({lds_ks}/{lds_sigma})')
            smoothed_value = convolve1d(
                np.asarray([v for _, v in value_dict.items()]), weights=lds_kernel_window, mode='constant')
            num_per_label = [smoothed_value[min(max_target - 1, int(label))] for label in labels]

        weights = [np.float32(1 / x) for x in num_per_label]
        scaling = len(weights) / np.sum(weights)
        weights = [scaling * x for x in weights]
        return weights

    def __getslabels__(self, max_target=1, lds_kernel='gaussian', lds_ks=5, lds_sigma=2):
        value_dict = {x: 0 for x in np.round(np.linspace(0,max_target,11),1)}
        labels = self.targets
        for label in labels:
            value_dict[min(max_target, label)] += 1
        lds_kernel_window = get_lds_kernel_window(lds_kernel, lds_ks, lds_sigma)
        smoothed_value = convolve1d(
                np.asarray([v for _, v in value_dict.items()]), weights=lds_kernel_window, mode='constant')
        return smoothed_value,  np.asarray([v for _, v in value_dict.items()])