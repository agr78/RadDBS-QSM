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

fig, ax = plt.subplots(1,1)


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
