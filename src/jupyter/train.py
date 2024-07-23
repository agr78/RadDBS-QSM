# Import libraries
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import nibabel as nib
from sklearn.svm import SVR
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score, roc_curve, auc
import SimpleITK as sitk
import six
from radiomics import featureextractor 
import numpy as np
import os
import pickle
import pandas as pd
import logging
from scipy.stats import linregress
from sklearn.linear_model import QuantileRegressor
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import r_regression
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import RANSACRegressor
import os
import sys
sys.path.append('../')
import matplotlib.pyplot as plt
from pylab import rcParams
import pandas as pd
import warnings
from sklearnex import patch_sklearn, config_context
from sklearn.cluster import DBSCAN
from sklearn.exceptions import ConvergenceWarning
from sklearn import preprocessing as skp
from sklearn import model_selection as sms
from sklearn import feature_selection as skf
from sklearn import linear_model as slm
import numpy as np
import scipy.stats as stats
from IPython.display import HTML
import util as util
import nibabel as nib
import os
import pickle
from torch import nn
patch_sklearn()

def train_estimator(subsc,X_all_c,K_all_c,per_change,pre_updrs_off,aug,reg,save,rs0,verbose):
  results = np.zeros_like(per_change)
  K_nz = []
  for j in np.arange(len(subsc)):
        test_id = subsc[j]
        test_index = subsc == test_id
        train_index = subsc != test_id
        X_train = X_all_c[train_index,:,:]
        X_test = X_all_c[test_index,:,:]
        y_train = per_change[train_index]

        y_cat = y_train <= 0.3
        idy = np.where(y_cat==1)
        # Cross validation
                                              
        X0_ss0,_,X_test_ss0 = util.model_scale(skp.StandardScaler(),
                                                    X_train,train_index,X_test,
                                                    test_index,pre_updrs_off,None,None,None,None,None,None,None,None,None,False,False,False)
        cvn = 5
        cv_scores = np.zeros((cvn,1))
        rcfs = 1000
        rs = rs0
        (mu, sigma) = stats.norm.fit(y_train)
        kappa = stats.skew(y_train)
        print('Label distribution of:',mu,sigma,kappa)
        for jj in np.arange(2,cvn):
          # Resample to avoid stratification errors
          while np.sum(y_cat) < cvn:
            np.random.seed(rs)
            idyr = np.random.choice(np.asarray(idy).ravel())
            X0_ss0 = np.append(X0_ss0,X0_ss0[idyr,:].reshape(1,-1),axis=0)
            y_train = np.append(y_train,y_train[idyr])
            y_cat = y_train <= 0.3
            rs = rs+1
            print('Resampled to size',y_train.shape)
          if aug == 'nc':
            y_train_n = y_train+(1.96*sigma)*np.random.normal(0,1,1)
            y_train = np.hstack((y_train,y_train_n))
            y_cat = y_train <= 0.3
            X0_ss0 = np.vstack((X0_ss0,X0_ss0))
          if aug == 'wbs':
            lm0 = slm.LassoLarsCV(max_iter=1000,cv=jj,n_jobs=-1,normalize=False,eps=0.1)
            lm0.fit(X0_ss0,y_train)
            eps = y_train-lm0.predict(X0_ss0)
            eps_v = eps*np.random.normal(0,1,1)
            while len(eps_v) < len(y_train):
              eps_v = np.hstack((eps_v,eps*np.random.normal(0,1,1)))
            y_train_n = y_train+eps_v
            y_train = np.hstack((y_train,y_train_n))
            y_cat = y_train <= 0.3
            X0_ss0 = np.vstack((X0_ss0,X0_ss0))
        
        for jj in np.arange(2,cvn):
          skf_g = sms.StratifiedKFold(n_splits=jj,shuffle=True,random_state=0)
          skf_gen = skf_g.split(X0_ss0,y_cat)
          if reg == True:
            est_type = 'ls'
            lm = slm.LassoLarsCV(max_iter=1000,cv=jj,n_jobs=-1,normalize=False,eps=0.1)
          else:
            est_type = 'lr'
            lm = slm.LogisticRegressionCV(n_jobs=-1,cv=jj,class_weight=None,penalty='l1',solver='liblinear',random_state=0)
            y_train = y_train > 0.3
          with warnings.catch_warnings() and np.errstate(divide='ignore', invalid='ignore'):
            # Feature selection
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            sel = skf.RFECV(lm,step=rcfs,cv=skf_gen,n_jobs=1)
            X0_ss = sel.fit_transform(X0_ss0,y_train)
            est = lm.fit(X0_ss,y_train)
            cv_scores[jj] = est.score(X0_ss,y_train)
            print('LassoCV score for',jj,'is',cv_scores[jj],'from dataset of size',X0_ss.shape)
            
        with warnings.catch_warnings() and np.errstate(divide='ignore', invalid='ignore'):        
          best_cv = np.argmax(cv_scores)
          # Break any ties
          if np.sum(cv_scores == best_cv) > 1:
            cv_scores_tb = np.zeros((np.sum(cv_scores == best_cv),1))
            for jjj in (cv_scores == cv_scores[best_cv]):
              if jjj > 0:
                print('Breaking tie')
                skf_g = sms.StratifiedKFold(n_splits=np.arange(2,cvn)[jjj],shuffle=True,random_state=1)
                skf_gen = skf_g.split(X0_ss0,y_cat) 
                X0_ss = sel.fit_transform(X0_ss0,y_train)
                if reg == True:
                    lm = slm.LassoLarsCV(max_iter=1000,cv=jj,n_jobs=-1,normalize=False,eps=0.1)
                else:
                    lm = slm.LogisticRegressionCV(n_jobs=-1,cv=jj,class_weight=None,penalty='l1',solver='liblinear',random_state=1)
                est = lm.fit(X0_ss,y_train)
                cv_scores_tb[jjj] = lm.score(X0_ss,y_train)
            best_cv = np.argmax(cv_scores_tb)
  
          # Fit whole dataset with optimal cv
          if reg == True:
            lm = slm.LassoLarsCV(max_iter=1000,cv=best_cv,n_jobs=-1,normalize=False,eps=0.1)
          else:
            lm = slm.LogisticRegressionCV(n_jobs=-1,cv=best_cv,class_weight=None,penalty='l1',solver='liblinear',random_state=0)
          sel = skf.RFECV(lm,step=rcfs,cv=best_cv)
          X0_ss = sel.fit_transform(X0_ss0,y_train)
          X_test_ss = sel.transform(X_test_ss0)
          K_ss = sel.transform(K_all_c.reshape(1,-1))

        # LASSO
        with warnings.catch_warnings():
          warnings.filterwarnings("ignore", category=ConvergenceWarning)
          if reg == True:
            lm = slm.LassoLarsCV(max_iter=1000,cv=best_cv,n_jobs=-1,normalize=False,eps=0.1)
            est = lm.fit(X0_ss,y_train)
            results[j] = lm.predict(X_test_ss)
          else:
            lm = slm.LogisticRegressionCV(n_jobs=-1,cv=best_cv,class_weight=None,penalty='l1',solver='liblinear',random_state=1)
            est = lm.fit(X0_ss,y_train)
            results[j,0] = lm.predict(X_test)
            results[j,1] = lm.predict_proba(X_test)[0][0]
    
        if verbose == True:
            print('Estimator predicts',str(np.round(results[j],4)),
                    'for case with',str(np.round((per_change)[j],2)),'and selected CV',best_cv,'and',sum(y_cat),'minority cases')
        K_nz.append(np.squeeze(K_ss)[abs(lm.coef_)>0])
  if save == True:
      np.save('results'+'_'+str(est_type)+'_'+str(aug)+'_'+str(rs0)+'.npy',results)
      np.save('features'+'_'+str(est_type)+'_'+str(aug)+'_'+str(rs0)+'.npy',K_nz)
      np.save('coefs'+'_'+str(est_type)+'_'+str(aug)+'_'+str(rs0)+'.npy',lm.coef_[abs(lm.coef_)>0])

  return results