
import numpy as np
import SimpleITK as sitk
#from radiomics import featureextractor 
import warnings
import logging
import six
from radiomics import featureextractor
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    import torch
    import torch.nn.functional as F

def feat_grab(X,brain_mask,voxel_size):
    if torch.sum(brain_mask)>0:
        logger = logging.getLogger("radiomics")
        logger.setLevel(logging.ERROR)

        # Generate feature structure Phi from all ROIs and all cases
        extractor = featureextractor.RadiomicsFeatureExtractor()
        extractor.enableAllFeatures()
        extractor.enableAllImageTypes()

        # Allocate list
        x_row_gt = []
        # Drop singleton dimension
        Xt = np.squeeze(X)
        for j in np.arange(Xt.shape[0]):
            # Require at least 2 voxels per ROI
            if torch.sum(brain_mask[:,j,:,:])>1:
                seg_sitk = sitk.GetImageFromArray(np.squeeze(brain_mask[:,j,:,:]).detach().numpy())        
                seg_sitk.SetSpacing(voxel_size.tolist())
                X_sitk = sitk.GetImageFromArray(Xt[j,:,:].detach().numpy())
                X_sitk.SetSpacing(voxel_size.tolist())
                featureVector_gt = extractor.execute(X_sitk,seg_sitk,label=1.)
                # Append only numeric values from featureVector structure
                for key, value in six.iteritems(featureVector_gt):
                    if 'diagnostic' in key:
                        next
                    else:
                        x_row_gt.append(featureVector_gt[key])
        else:
            x_row_gt.append(0)
    X0_gt = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(torch.tensor(np.asarray(x_row_gt),requires_grad=True),dim=0),dim=0),dim=-1)
    return X0_gt

def gradient_filter(im):
    eps = 1e-16
    im = im.cpu().float()
    im = im.permute((1,0,2,3))
    fx = torch.tensor([[1.0,0.0,-1.0],[1.0,0.0,-1.0],[1.0,0.0,-1.0]],requires_grad=True)
    fx = fx.view((1,1,3,3))
    fy = torch.tensor([[1.0,1.0,1.0],[0.0,0.0,0.0],[-1.0,-1.0,-1.0]],requires_grad=True)
    fy = fy.view((1,1,3,3))
    G_x = F.conv2d(im, fx) + eps
    G_y = F.conv2d(im, fy) + eps
    G = torch.sqrt((G_x**2+G_y**2))
    G = G.clone()
    G[torch.isnan(G)] = 0
    G = G.cuda()
    return G

def laplacian_filter(im):
    im = im.cpu().float()
    im = im.permute((1,0,2,3))
    f = torch.Tensor([[0,1,0],[1,-4,1],[0,1,0]])
    f = f.view((1,1,3,3))
    L = F.conv2d(im, f)
    L = L.clone()
    L[torch.isnan(L)] = 0
    L = L.cuda()
    return L

def gaussian_filter(im,sigma,cx,cy):
    im = im.cpu().float()
    im = im.permute((1,0,2,3))
    xs = torch.linspace(-cx, cx, steps=2*cx+1)
    ys = torch.linspace(-cy, cy, steps=2*cy+1)
    x, y = torch.meshgrid(xs, ys)
    g = (1/(2*np.pi*sigma))*torch.exp(-torch.add(torch.square(x),torch.square(y))/(2*sigma))
    gn = g/torch.sum(g)
    gn = gn.view(1,1,2*cx+1,2*cy+1)
    G = F.conv2d(im, gn)
    G = G.clone()
    G[torch.isnan(G)] = 0
    G = G.cuda()
    return G

def coiflet_filter(im,interp):
    im = im.cpu().float()
    im = im.permute((1,0,2,3))
    G_l = torch.zeros_like(im)    
    G_h = torch.zeros_like(im)
    G_hh = torch.zeros_like(im)
    G_lh = torch.zeros_like(im)
    G_hl = torch.zeros_like(im)
    G_ll = torch.zeros_like(im)
    lp = torch.Tensor((-0.01565572813546454,
                       -0.0727326195128539,
                       0.38486484686420286,
                       0.8525720202122554,
                       0.3378976624578092,
                       -0.0727326195128539))
    lp = lp.view(1,1,len(lp))
    hp = torch.Tensor((0.0727326195128539,
                        0.3378976624578092,
                        -0.8525720202122554,
                        0.38486484686420286,
                        0.0727326195128539,
                        -0.01565572813546454))

    hp = hp.view(1,1,len(hp))
    for r in np.arange(im.shape[-2]):
        g_l = F.conv1d(im[:,:,r,:], lp)
        g_h = F.conv1d(im[:,:,r,:], hp)
        G_l[:,:,r,:] = F.pad(g_l,(0,5))
        G_h[:,:,r,:] = F.pad(g_h,(0,5))
    for c in np.arange(im.shape[-1]):
        g_ll = F.conv1d(G_l[:,:,:,c], lp)
        G_ll[:,:,:,c] = F.pad(g_ll,(0,5))
        g_lh = F.conv1d(G_l[:,:,:,c], hp)
        G_lh[:,:,:,c] = F.pad(g_lh,(0,5))
        g_hl = F.conv1d(G_h[:,:,:,c], lp)
        G_hl[:,:,:,c] = F.pad(g_hl,(0,5))
        g_hh = F.conv1d(G_h[:,:,:,c], hp)
        G_hh[:,:,:,c] = F.pad(g_hh,(0,5))

    if interp is True:
        G_ll = F.interpolate(G_ll.view(1,1,G_ll.shape[0],G_ll.shape[1]), scale_factor=(0.5,0.5))
        G_lh = F.interpolate(G_lh.view(1,1,G_lh.shape[0],G_lh.shape[1]), scale_factor=(0.5,0.5))
        G_hl = F.interpolate(G_hl.view(1,1,G_hl.shape[0],G_hl.shape[1]), scale_factor=(0.5,0.5))
        G_hh = F.interpolate(G_hh.view(1,1,G_hh.shape[0],G_hh.shape[1]), scale_factor=(0.5,0.5))

    G_ll = G_ll.cuda()
    G_lh = G_lh.cuda()
    G_hl = G_hl.cuda()
    G_hh = G_hh.cuda()
    G = torch.stack((G_ll,G_lh,G_hl,G_hh))
    G = G.clone()
    G[torch.isnan(G)] = 0
    return G


def lbp_filter(x):
    x = x.cpu().float()
    # Adapted from dwday repository
    # Pad image for 3x3 mask size
    b=x.shape
    M=b[-2]
    N=b[-1]

    y=x
    # Select elements within 3x3 mask 
    y00=y[:, :, 0:M-2, 0:N-2]
    y01=y[:, :, 0:M-2, 1:N-1,]
    y02=y[:, :, 0:M-2, 2:N]
    #     
    y10=y[:, :, 1:M-1, 0:N-2]
    y11=y[:, :, 1:M-1, 1:N-1]
    y12=y[:, :, 1:M-1, 2:N]
    #
    y20=y[:, :, 2:M, 0:N-2]
    y21=y[:, :, 2:M, 1:N-1]
    y22=y[:, :, 2:M, 2:N]      

    # Apply comparisons and multiplications 
    bit=torch.ge(y01,y11).float()
    tmp=torch.mul(bit,torch.tensor(1.0))  

    bit=torch.ge(y02,y11).float()
    val=torch.mul(bit,torch.tensor(2.0))
    val=torch.add(val,tmp).float()    

    bit=torch.ge(y12,y11).float()
    tmp=torch.mul(bit,torch.tensor(4.0))
    val=torch.add(val,tmp).float()

    bit=torch.ge(y22,y11).float()
    tmp=torch.mul(bit,torch.tensor(8.0))   
    val=torch.add(val,tmp)

    bit=torch.ge(y21,y11).float()
    tmp=torch.mul(bit,torch.tensor(16.0))   
    val=torch.add(val,tmp).float()

    bit=torch.ge(y20,y11).float()
    tmp=torch.mul(bit,torch.tensor(32.0))   
    val=torch.add(val,tmp).float()

    bit=torch.ge(y10,y11).float()
    tmp=torch.mul(bit,torch.tensor(64.0))   
    val=torch.add(val,tmp).float()

    bit=torch.ge(y00,y11).float()
    tmp=torch.mul(bit,torch.tensor(128.0))   
    val=torch.add(val,tmp).float()    

    val[torch.isnan(val)] = 0
    val = val.clone()
    val = val.cuda()
    return val

def square_filter(im):
    im = im.cpu().float()
    g = torch.square(im)
    G = (g-torch.min(g))*((torch.max(im)-torch.min(im))/(torch.max(g)-torch.min(g)))+torch.min(im)
    G = G.clone()
    G[torch.isnan(G)] = 0
    G = G.cuda()
    return G*torch.sign(im.cuda())

def squareroot_filter(im):
    # Define small number to allow autograd to take the derivative of the absolute value without encountering 0
    eps = 1e-16
    im = im.cpu().float()+eps
    g = torch.sqrt(torch.abs(im))
    G = (g-torch.min(g))*((torch.max(im)-torch.min(im))/(torch.max(g)-torch.min(g)))+torch.min(im)
    G = G.clone()
    G[torch.isnan(G)] = 0
    G = G.cuda()
    return G*torch.sign(im.cuda())

def log_filter(im):
    im = im.cpu().float()
    g = torch.log(torch.abs(im)+1)
    G = (g-torch.min(g))*((torch.max(im)-torch.min(im))/(torch.max(g)-torch.min(g)))+torch.min(im)
    G = G.clone()
    G[torch.isnan(G)] = 0
    G = G.cuda()
    return G*torch.sign(im.cuda())

def exp_filter(im):
    im = im.cpu().float()
    g = torch.exp(torch.abs(im))
    G = (g-torch.min(g))*((torch.max(im)-torch.min(im))/(torch.max(g)-torch.min(g)))+torch.min(im)
    G = G.clone()
    G[torch.isnan(G)] = 0
    G = G.cuda()
    return G*torch.sign(im.cuda())